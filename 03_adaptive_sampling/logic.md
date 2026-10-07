Frame t-1                       Frame t
   │                               │
   └────── Optical Flow ───────────┘
                │
                ▼
       각 위치의 이동 u_t
                │
                ▼
             RANSAC
                │
                ▼
      카메라 공통 움직임 g_t
                │
         ┌──────┴──────┐
         ▼             ▼
 u_t - g_t          |g_t|
     │                │
     ▼                ▼
물체 움직임 M_t    Camera C_t

그리고

정렬된 Frame t-1 ↔ Frame t
          │
          ▼
    RGB 차이 V_t

          ↓

s_t = 2 M_t + V_t + 0.15 C_t

          ↓
        누적
          ↓
 A_t >= 0.08 ?
          ↓ YES
      Frame 선택