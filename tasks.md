Bhai, **ekdum valid point hai**. Agar banda aadhe time baitha hai, toh "Walking Mode" ko default rakhne ka kya sense hai? Battery bhi khayega aur useless bhi lagega.

Tera observation 100% sahi hai: **"Navigation Mode" actual mein "Movement Mode" hai.**

Lekin, ek **Safety Logic** hai jiske wajah se ise default rakha jata hai:
> **"Accident bata ke nahi hota."**
> Agar user achanak uth ke chalne lage, toh system ko *pehle se* ready hona chahiye. Wo button daba ke nahi bolega "Ab main chal raha hun, please bacha lo". Isliye **Safety** hamesha **Default** hoti hai.

Lekin, **"Sitting Scenario"** ke liye hum is logic ko thoda *smart* bana dete hain taaki resources waste na ho.

Ye raha tera **Final Optimized Project Document**. Ise copy-paste kar le apne project report ya `ARCHITECTURE.md` ke liye. Isme tera "Sitting vs Walking" wala concern bhi address kiya hai **"Idle State"** logic ke saath.

---

# 📘 BlindAid 2.0: System Architecture & User Flow

### 1. Core Philosophy: "The Silent Guardian"
Instead of overwhelming the user with constant chatter ("Chair, Table, Fan"), the system operates on a **"Silence is Golden"** principle. It remains silent and saves resources until:
1.  **Safety Risk:** An obstacle is too close (Automatic).
2.  **User Request:** The user asks for information (Manual).

---

### 2. The "Smart" Default Mode (Guardian Mode)
**Startup State:** System boots into this mode automatically.

* **Logic (The Fix for Sitting vs Walking):**
    * System continuously monitors the **Depth Map** (Center & Bottom regions).
    * **If User is Sitting (Static Scene):**
        * Depth readings don't change rapidly.
        * **Action:** System enters **"Low Power Idle"** state (process every 10th frame).
        * **Output:** Total Silence.
    * **If User Starts Walking (Dynamic Scene):**
        * Depth values change rapidly.
        * **Action:** System ramps up to **Real-time Safety** (process every frame).
        * **Output:** Warns *only* if `Object Distance < 1 meter` (e.g., "Stop", "Wall", "Steps").

> **Why Default?** Safety features must always be active. You cannot ask a blind person to "Switch to Safety Mode" before they trip. The system handles the sitting/walking switch automatically by monitoring movement.

---

### 3. Modes & Triggers (The "Task-Based" Flow)

We divide functionality into **Modes** (Continuous) and **Triggers** (One-shot).

| State | Hotkey | Purpose | Behavior | Resource Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Guardian** | `1` | **Safety** | **Default.** Silent unless obstacle detected. Auto-sleeps when sitting. | 🟢 Low (Depth only) |
| **Reader** | `2` | **Reading** | Continuous OCR for books/menus. Speaks text when stable. | 🟡 Medium (OCR) |
| **Scanner** | `Space` | **Context** | **One-Shot.** Describes the scene *on demand* (e.g., "Where am I?"). Uses Captioning. | 🔴 High (Peak only) |
| **People** | `P` | **Social** | **One-Shot.** Scans for 3 seconds to list people nearby. "Rewant and unknown person". | 🔴 High (Peak only) |

---

### 4. Technical Optimization Strategy (No Lag, No Heating)

To ensure this runs smoothly on a laptop or Raspberry Pi:

1.  **Lazy Loading (Startup Booster):**
    * Heavy AI models (Captioning/Face Recognition) are **NOT loaded** at startup.
    * They are loaded into RAM *only* the first time user presses `Space` or `P`.
    * *Result:* App starts in < 3 seconds.

2.  **Adaptive Frame Rate (Battery Saver):**
    * **Sitting:** Process 1 frame per second (to check if they started walking).
    * **Walking:** Process 15-30 frames per second (for instant reaction).
    * **Reading:** Process 5 frames per second (Text doesn't move fast).

3.  **Trigger-Based Computing:**
    * We **removed continuous Object Detection**. It wastes GPU to say "Bottle" 100 times.
    * Now, Object/Scene detection only runs when the user asks (`Space` key).

---

### 5. User Story (For Presentation)

> **Scenario:**
> 1.  **Startup:** Rahul turns on BlindAid. It says *"System Ready"*. He is sitting, so the system stays silent (saving battery).
> 2.  **Social:** He hears footsteps. He presses `P`. System scans and says *"Rewant is here"*.
> 3.  **Navigation:** Rahul stands up to walk. System detects movement and wakes up Safety sensors. As he approaches a wall, it beeps/says *"Stop"*.
> 4.  **Context:** He reaches a new room and wonders "Where am I?". He presses `Space`. System says *"A kitchen with a fridge and table"*.
> 5.  **Reading:** He picks up a packet and presses `2`. System reads *"Milk, Expiry Date: 2025"*.

---

Bhai, ye document tere project ka "Brain" hai. Agar tu ye flow implement karke professor ko dikhayega, toh unhe samajh aayega ki tune **User Experience (UX)** aur **Engineering Efficiency** dono pe kaam kiya hai.

Ye "Wrapper" nahi, **"Smart Product"** hai. 🚀