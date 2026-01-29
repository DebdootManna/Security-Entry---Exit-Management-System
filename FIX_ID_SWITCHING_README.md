# Fix for Over-Segmentation / ID-Switching Problem

## Problem Statement

**Current Issue:** The system reports 38 unique visitors when only 2 physical people are present in the video stream. This is a classic "ID-switching" or "over-segmentation" problem where the system creates new UUIDs for the same person repeatedly instead of maintaining persistent tracking.

**Root Causes Identified:**
1. **Threshold too strict:** 0.6 cosine distance threshold is too strict for varying lighting conditions in museum galleries
2. **No temporal smoothing:** System creates new UUID immediately when a match fails
3. **No grace period:** If a person is briefly occluded (turns around, face not visible for 2-3 seconds), they get a new UUID when they reappear
4. **No body/clothing features:** System relies only on face embeddings, no fallback when face is unavailable
5. **Frame-by-frame independence:** Each frame is processed independently with no concept of "this person was here 1 second ago"

---

## Solution: Temporal Smoothing with Grace Period

### Key Changes to Implement

#### 1. **Relaxed Thresholds for Museum Lighting**

```python
# OLD (too strict)
SIMILARITY_THRESHOLD = 0.6
DETECTION_CONFIDENCE = 0.8

# NEW (adjusted for real-world conditions)
FACE_THRESHOLD_STRICT = 0.55      # High-confidence matches
FACE_THRESHOLD_NORMAL = 0.70      # Standard matching (relaxed from 0.6)
FACE_THRESHOLD_RELAXED = 0.80     # For temporal continuity
DETECTION_CONFIDENCE = 0.70       # Lowered from 0.8
```

**Why:** Museum galleries have varying lighting (windows, spotlights, shadows). A strict 0.6 threshold causes false negatives (same person not recognized due to lighting change).

---

#### 2. **Temporal Smoothing - Grace Period**

```python
GRACE_PERIOD_SECONDS = 3.0  # Wait 3 seconds before creating new ID
MAX_SESSION_GAP_SECONDS = 5.0  # Max gap to maintain session
TEMPORAL_WINDOW_SIZE = 5  # Average last 5 embeddings
```

**Implementation:**
- **Don't create new UUID immediately** when a match fails
- **Wait 2-3 seconds** to see if the person reappears
- **Check recently lost visitors first** before declaring "new person"
- **Maintain "Inside_Now" state** with active sessions and recently-lost sessions

**Example Flow:**
```
Frame 1: Person detected → UUID-001 created
Frame 50: Person turns around (no face) → UUID-001 marked as "lost" (not deleted!)
Frame 55: Person turns back → System checks "lost" visitors first → Matches UUID-001 → Re-acquired!
```

---

#### 3. **Multi-Stage Matching Strategy**

Instead of single-pass matching, use a cascade:

**Stage 1: Check Recently Lost Visitors (Within Grace Period)**
```python
if TrackingConfig.CHECK_LOST_VISITORS_FIRST:
    lost_match = check_lost_visitors(face_embedding)
    if lost_match:
        return lost_match  # Recovered!
```

**Stage 2: Check Active Sessions (People Currently Visible)**
```python
active_match = check_active_sessions(face_embedding)
if active_match:
    return active_match
```

**Stage 3: Try Body Features if Face Unavailable**
```python
if face_embedding is None and body_img is not None:
    body_match = check_body_features(body_img)
    if body_match:
        return body_match  # Matched using clothing!
```

**Stage 4: Check Grace Period Before Creating New UUID**
```python
if has_recent_unmatched_sessions():
    # Someone was just lost - be more lenient
    lenient_match = check_active_sessions(
        face_embedding, 
        threshold=RELAXED_THRESHOLD  # 0.80 instead of 0.70
    )
    if lenient_match:
        return lenient_match
```

**Stage 5: Create New Visitor (Only After All Checks)**
```python
return create_new_visitor(face_embedding)
```

---

#### 4. **Body/Clothing Features as Fallback**

When face is unavailable (person turned around, occluded), use body features:

```python
def extract_body_features(body_img):
    """Extract clothing color and texture features"""
    hsv = cv2.cvtColor(body_img, cv2.COLOR_BGR2HSV)
    
    # Color histograms (clothing color)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.cvtColor([hsv], [2], None, [256], [0, 256])
    
    # Edge density (texture)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    return {
        'hist_h': hist_h,
        'hist_s': hist_s,
        'hist_v': hist_v,
        'edge_density': edge_density
    }
```

**Body Matching Threshold:** 0.65-0.75 (more lenient than face since clothing is less unique)

---

#### 5. **"Inside_Now" State Management**

Maintain two tables:

**Active Sessions (Currently Visible):**
```python
active_sessions = {
    'uuid-001': VisitorSession(
        visitor_id='uuid-001',
        first_seen=datetime(...),
        last_seen=datetime(...),  # Updated every frame matched
        face_embeddings=deque([emb1, emb2, emb3, emb4, emb5]),  # Last 5
        body_features=deque([...]),
        encounter_count=15,
        is_lost=False
    )
}
```

**Lost Sessions (Within Grace Period):**
```python
lost_sessions = {
    'uuid-002': VisitorSession(
        visitor_id='uuid-002',
        last_seen=datetime(...),  # 2 seconds ago
        lost_since=datetime(...),  # When tracking was lost
        face_embeddings=deque([...]),
        is_lost=True
    )
}
```

**Session Lifecycle:**
```
NEW DETECTION
    ↓
[Active Sessions] ← Match success → Update session
    ↓ (no match for 1-2 seconds)
[Lost Sessions] ← Grace period (2-3 seconds) → Check first on new detection
    ↓ (grace period expired)
[Historical Archive] ← Long-term storage
```

---

### Implementation Steps

#### Step 1: Update `config.py`

```python
# Relaxed thresholds
FACE_THRESHOLD_NORMAL = 0.70  # Was 0.6
DETECTION_CONFIDENCE = 0.70   # Was 0.8

# Temporal settings
GRACE_PERIOD_SECONDS = 3.0
TEMPORAL_WINDOW_SIZE = 5
MAX_SESSION_GAP_SECONDS = 5.0

# Multi-stage matching
CHECK_LOST_VISITORS_FIRST = True
USE_BODY_FEATURES = True
ENABLE_MULTI_STAGE_MATCHING = True
```

---

#### Step 2: Refactor `check_in_visitor` Logic

**OLD (Immediate UUID Creation):**
```python
def check_in_visitor(face_embedding):
    match = database.search(face_embedding)
    if not match:
        visitor_id = create_new_uuid()  # ❌ Too aggressive!
        database.add(visitor_id, face_embedding)
    return visitor_id
```

**NEW (Multi-Stage with Grace Period):**
```python
def check_in_visitor(face_img, body_img, bbox):
    # Extract features
    face_embedding = extract_face_features(face_img)
    body_features = extract_body_features(body_img)
    
    # STAGE 1: Check recently lost visitors first
    if CHECK_LOST_VISITORS_FIRST:
        lost_match = check_lost_visitors(face_embedding, body_features)
        if lost_match:
            # Recovered a lost visitor!
            move_to_active(lost_match['visitor_id'])
            print(f"[✓ RECOVERED] {lost_match['visitor_id'][:8]}...")
            return lost_match
    
    # STAGE 2: Check active sessions
    active_match = check_active_sessions(face_embedding, body_features)
    if active_match:
        return active_match
    
    # STAGE 3: Body features if face unavailable
    if face_embedding is None and body_features is not None:
        body_match = check_body_features(body_features)
        if body_match:
            return body_match
    
    # STAGE 4: Check if any session just lost (grace period)
    if has_recent_unmatched_sessions():
        # Be more lenient - might be same person
        lenient_match = check_active_sessions(
            face_embedding, 
            threshold=FACE_THRESHOLD_RELAXED  # 0.80
        )
        if lenient_match:
            return lenient_match
    
    # STAGE 5: Create new visitor (all checks exhausted)
    visitor_id = str(uuid.uuid4())
    create_new_session(visitor_id, face_embedding, body_features)
    print(f"[★ NEW] {visitor_id[:8]}... (Total unique: {total_visitors})")
    return {'visitor_id': visitor_id, 'is_new': True}
```

---

#### Step 3: Add VisitorSession Class

```python
class VisitorSession:
    """Represents a single visitor with temporal tracking"""
    
    def __init__(self, visitor_id, face_embedding=None, body_features=None):
        self.visitor_id = visitor_id
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.encounter_count = 1
        
        # Temporal smoothing - store last N embeddings
        self.face_embeddings = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        self.body_features = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        
        if face_embedding is not None:
            self.face_embeddings.append(face_embedding)
        if body_features is not None:
            self.body_features.append(body_features)
        
        # Tracking state
        self.is_lost = False
        self.lost_since = None
        self.consecutive_misses = 0
    
    def update_match(self, face_embedding=None, body_features=None):
        """Update session with new observation"""
        self.last_seen = datetime.now()
        self.encounter_count += 1
        self.consecutive_misses = 0
        
        if self.is_lost:
            # Reacquired!
            lost_duration = (datetime.now() - self.lost_since).total_seconds()
            print(f"[REACQUIRED] {self.visitor_id[:8]} after {lost_duration:.1f}s")
        
        self.is_lost = False
        self.lost_since = None
        
        if face_embedding is not None:
            self.face_embeddings.append(face_embedding)
        if body_features is not None:
            self.body_features.append(body_features)
    
    def mark_lost(self):
        """Mark as lost (not matched in current frame)"""
        if not self.is_lost:
            self.is_lost = True
            self.lost_since = datetime.now()
        self.consecutive_misses += 1
    
    def seconds_lost(self):
        """Time since lost"""
        if not self.is_lost or self.lost_since is None:
            return 0.0
        return (datetime.now() - self.lost_since).total_seconds()
    
    def get_average_embedding(self):
        """Get temporally smoothed embedding"""
        if len(self.face_embeddings) == 0:
            return None
        # Average recent embeddings for stability
        avg = np.mean(list(self.face_embeddings), axis=0)
        return avg / np.linalg.norm(avg)  # Normalize
```

---

#### Step 4: Update Database Class

```python
class EnhancedVisitorDatabase:
    def __init__(self, face_threshold=0.70, grace_period=3.0):
        self.face_threshold = face_threshold
        self.grace_period = grace_period
        
        # Active sessions (Inside_Now)
        self.active_sessions = {}  # visitor_id -> VisitorSession
        
        # Recently lost (within grace period)
        self.lost_sessions = {}  # visitor_id -> VisitorSession
        
        # Historical (for analytics)
        self.historical_sessions = []
    
    def check_lost_visitors(self, face_embedding, body_features):
        """Check recently lost visitors first"""
        for visitor_id, session in list(self.lost_sessions.items()):
            # Check if still within grace period
            if session.seconds_lost() > self.grace_period:
                # Expired - move to historical
                self.historical_sessions.append(session)
                del self.lost_sessions[visitor_id]
                continue
            
            # Try to match
            avg_embedding = session.get_average_embedding()
            if avg_embedding is not None:
                distance = cosine(face_embedding, avg_embedding)
                
                # Use relaxed threshold for lost sessions
                if distance < FACE_THRESHOLD_RELAXED:
                    # Match! Move back to active
                    self.active_sessions[visitor_id] = session
                    del self.lost_sessions[visitor_id]
                    session.update_match(face_embedding, body_features)
                    return {'visitor_id': visitor_id, 'is_new': False}
        
        return None
    
    def mark_unmatched_as_lost(self, matched_ids):
        """Mark sessions not seen in current frame as lost"""
        for visitor_id in list(self.active_sessions.keys()):
            if visitor_id not in matched_ids:
                session = self.active_sessions[visitor_id]
                session.mark_lost()
                
                # Move to lost_sessions after threshold
                if session.consecutive_misses > 3:  # ~1 second at 30fps
                    self.lost_sessions[visitor_id] = session
                    del self.active_sessions[visitor_id]
                    print(f"[→ LOST] {visitor_id[:8]}...")
```

---

### Expected Results

**Before Fix:**
```
[NEW VISITOR] Created: 8f3a2b15... (Total unique: 1)
[NEW VISITOR] Created: 9c4d5e27... (Total unique: 2)  # Same person!
[NEW VISITOR] Created: 1a7b8c39... (Total unique: 3)  # Still same!
...
[NEW VISITOR] Created: 4f9e2d60... (Total unique: 38)  # Only 2 people!
```

**After Fix:**
```
[NEW VISITOR] Created: 8f3a2b15... (Total unique: 1)
[MATCH] ID: 8f3a2b15 | Score: 0.750 | Encounters: 2
[MATCH] ID: 8f3a2b15 | Score: 0.680 | Encounters: 3
[→ LOST] 8f3a2b15... (person turned around)
[✓ RECOVERED] 8f3a2b15... after 1.2s
[MATCH] ID: 8f3a2b15 | Score: 0.720 | Encounters: 4
[NEW VISITOR] Created: 2d9c4f81... (Total unique: 2)  # Second person
[MATCH] ID: 2d9c4f81 | Score: 0.810 | Encounters: 2
...
Final: Total Unique Visitors: 2  ✓ Correct!
```

---

### Testing the Fix

#### 1. Run with Original Code (Should Show Bug)
```bash
python face_detection_simple.py
# OR
python face_reidentification_test.py
```

Expected: 38 unique visitors for 2 people

#### 2. Apply Configuration Changes

Edit `face_detection_simple.py` or your tracking code:

```python
# Change line ~41
self.database = SimpleFaceDatabase(similarity_threshold=0.75)  # Was 0.70
```

#### 3. Add Temporal Tracking

The full fix requires implementing the `VisitorSession` class and multi-stage matching logic described above.

#### 4. Monitor Console Output

Look for these indicators:
- `[RECOVERY]` messages → Good! System is recovering lost tracks
- `[LENIENT MATCH]` → System using relaxed threshold successfully
- `[BODY MATCH]` → Fallback to clothing features working
- Ratio of `[NEW VISITOR]` to `[MATCH]` should be ~1:20 (not 38:0!)

---

### Quick Configuration-Only Fix

If you can't refactor the code immediately, try these threshold adjustments:

```python
# In face_detection_simple.py line ~41:
self.database = SimpleFaceDatabase(similarity_threshold=0.75)  # Increase from 0.70

# In face_reidentification_test.py line ~78:
self.database = FaceDatabase(
    use_faiss=True,
    dimension=512,
    similarity_threshold=0.75  # Increase from 0.6
)

# In config.py:
DETECTION_CONFIDENCE_THRESHOLD = 0.70  # Lower from 0.8
SIMILARITY_THRESHOLD = 0.75  # Increase from 0.6
```

This alone won't fully solve the problem but should reduce 38 IDs to ~5-10 IDs.

---

### Full Solution Implementation

For production deployment at Indian Museum & Victoria Memorial, I recommend:

1. **Implement the full temporal tracking system** with `VisitorSession` class
2. **Add body feature extraction** for occlusion handling
3. **Use multi-stage matching** as described
4. **Set thresholds:**
   - Face: 0.70 (normal), 0.80 (relaxed)
   - Body: 0.75
   - Grace period: 3 seconds
   - Detection confidence: 0.70

5. **Monitor and tune:**
   - Track re-identification rate: `successful_reids / (successful_reids + new_visitors)`
   - Target: >90% re-identification rate
   - Adjust thresholds based on actual museum lighting conditions

---

### Summary

The fix addresses the root cause: **premature UUID creation without temporal context**.

**Key Principle:** 
> "When in doubt, wait and check recently lost visitors before creating a new ID."

This transforms the system from a **frame-by-frame detector** into a true **visitor tracking system** with memory and temporal awareness.

**Expected Improvement:**
- 38 IDs → 2-3 IDs for 2 people
- ~95% reduction in false new-visitor declarations
- Robust handling of occlusions, lighting changes, and brief track losses

---

For questions or implementation help, refer to the Technical Specification document and the example implementations in this repository.