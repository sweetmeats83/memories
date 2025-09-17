# 📜 Memories App – Project Map

## **Purpose**

Memories is a web application designed to **capture, preserve, and organize personal stories** in various formats (text, audio, video, images).  
The goal is to make it easy for family members and friends to:

- **Receive prompts** that inspire storytelling.
- **Respond** with recordings, text, or uploaded media.
- **Edit and refine transcriptions** generated automatically from audio/video.
- **Explore completed memories** in a visually appealing timeline or “storybook” format.
- **Generate physical keepsakes** such as PDFs or printed books with QR codes linking to multimedia responses.

Admins can manage users, create and assign prompts, view progress, and curate the growing memory archive.

---

## **Core Interfaces**

1. **Admin Dashboard** (`admin_dashboard.html`)
   - Manage users and invitations.
   - Create, edit (inline or modal), delete prompts (text + media attachments).
   - Organize prompts into **chapters** (collapsible sections).
   - Assign prompts to users.
   - View response activity and progress reports.
   - Search prompts by text or tags.

2. **User Dashboard** (`user_dashboard.html`)
   - See “Prompt of the Week” prominently displayed.
   - Browse assigned prompts.
   - View completed responses as **floating thumbnails**.
   - Search memories by tags, keywords, or chapter.
   - Open a formatted “storybook view” for each response (with attached media and transcription).
   - Edit or delete responses.

3. **User Record Page** (`user_record.html`) *(future)*
   - Record audio/video directly in the browser.
   - Upload supporting media.
   - Wait for automatic transcription, then edit the generated text.
   - Save and attach all media to the response.
   - Auto-format responses via an LLM for readability.

---

## **Key Features**

- **Authentication:** FastAPI-Users-based auth with email, password, and admin roles.
- **Prompt Management:** Multimedia prompts grouped by chapters, editable inline or via modal.
- **Media Storage:** Files saved under `/static/uploads` with persistent Docker volume.
- **Transcription Service (Future):** Automated transcription using a local or cloud model (e.g., Whisper).
- **AI Text Processing (Future):** LLM formatting of transcripts to improve narrative flow.
- **PDF Generation (Future):** Compile stories into printable formats, embedding QR codes linking to audio/video responses.
- **Search & Tags:** Users and admins can search by tags or content keywords.
- **Mobile Friendly:** The UI is touch-optimized for tablets and mobile browsers.
- **Notifications:** Email or in-app alerts for new prompts and invitations.

---

## **Tech Stack**

- **Backend:** FastAPI (Python 3.11), SQLAlchemy (Async), PostgreSQL
- **Frontend:** Jinja2 templates, TailwindCSS, Vanilla JS
- **Authentication:** FastAPI-Users
- **Containerization:** Docker, Docker Compose
- **Media Storage:** Static volume-mounted directory (`./static/uploads`)
- **Planned AI:** Local transcription (Whisper), optional LLM text formatting
- **Future Search Enhancements:** Full-text search (PostgreSQL `tsvector`) for improved filtering.

---

## **File Structure Overview**

memories3/
├── app/
│ ├── main.py # FastAPI app initialization
│ ├── models.py # SQLAlchemy ORM models
│ ├── schemas.py # Pydantic schemas
│ ├── routes.py # Application routes (admin, users, prompts)
│ ├── utils.py # Helper functions
│ └── database.py # Database connection/session
├── static/
│ └── uploads/ # Media files stored here (volume-mounted)
├── templates/
│ ├── base.html
│ ├── admin_dashboard.html
│ ├── prompt_list.html
│ ├── admin_edit_prompt.html
│ └── (future user pages)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── PROJECT_MAP.md # 📌 This file


---

## **Planned Enhancements**

- FINALIZE THE ADMIN VIEW:
  1. Complete inline editing for prompts.
  2. Refine UI/UX (hover icons, collapsible chapters).
  3. Add Tag system for prompts and responses.
  4. Implement search system incorporating tags and keywords.
  5. Refine invitations and onboarding flow with role toggles.
  6. Fix logout/session issues.
  7. Add stats and user progress visualization with interactions on story view.

- **User Dashboard:** Pinterest-like floating tiles with organization and search filters.
- **User Story Creation:** Allow users to create their own stories outside prompts with chapter assignment and tags.
- **Transcription and AI Formatting:** Local Whisper + LLM-based formatting for final readable stories.
- **Offline & Export Features:** Allow downloading memories and entire chapters as PDFs with embedded QR codes.
- **Notification system:** Email or push notifications for new prompts and responses.
- **Accessibility Enhancements:** Ensure compatibility with screen readers and simplified navigation for older users.
