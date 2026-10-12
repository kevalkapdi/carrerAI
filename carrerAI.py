"""CareerAI authentication module.

Run the UI with: streamlit run carrerAI.py
"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import smtplib
from typing import Any, Iterator
from urllib import error as urllib_error
from urllib.parse import urlsplit
from urllib import request as urllib_request

import bcrypt
import pandas as pd
import pymupdf as fitz
import plotly.express as px
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.getenv("CAREERAI_DATABASE_PATH", PROJECT_ROOT / "data" / "careerai.db"))
ADMIN_EMAIL = os.getenv("CAREERAI_ADMIN_EMAIL", "kevalkapdi026@gmail.com").strip().casefold()
CAREERAI_CONFIG = {
	"name": "CareerAI",
	"version": os.getenv("CAREERAI_VERSION", "1.0.0").strip() or "1.0.0",
	"release_status": os.getenv("CAREERAI_RELEASE_STATUS", "Development").strip() or "Development",
	"last_updated": os.getenv("CAREERAI_LAST_UPDATED", date.today().isoformat()).strip(),
	"description": (
		"An intelligent career and resume platform that helps students and job seekers "
		"understand their skills, explore career paths, and plan practical next steps."
	),
	"project_type": "AI-powered Career Development Platform",
	"developer_name": "Keval Kapdi T.",
	"department": "Computer Engineering",
	"enrollment_number": "En-246020307101",
	"contact_email": os.getenv("CAREERAI_CONTACT_EMAIL", "kevalkapdi026@gmail.com").strip(),
	"contact_phone": os.getenv("CAREERAI_CONTACT_PHONE", "9909022098").strip(),
	"location": os.getenv("CAREERAI_LOCATION", "").strip(),
	"github_url": os.getenv("CAREERAI_GITHUB_URL", "https://github.com/kevalkapdi").strip(),
	"linkedin_url": os.getenv(
		"CAREERAI_LINKEDIN_URL",
		"https://www.linkedin.com/in/keval-kapdi?utm_source=share_via&utm_content=profile&utm_medium=member_android",
	).strip(),
	"support_information": os.getenv(
		"CAREERAI_SUPPORT_INFORMATION",
		"Use the Help & Support section below for common workflows and troubleshooting.",
	).strip(),
}
RESET_TOKEN_MINUTES = int(os.getenv("CAREERAI_RESET_TOKEN_MINUTES", "30"))
VERIFICATION_TOKEN_HOURS = int(os.getenv("CAREERAI_VERIFICATION_TOKEN_HOURS", "24"))
EXPOSE_DEV_TOKENS = os.getenv("CAREERAI_EXPOSE_DEV_TOKENS", "true").lower() == "true"
RESUME_STORAGE = Path(os.getenv("CAREERAI_RESUME_STORAGE", PROJECT_ROOT / "data" / "private_resumes"))
REPORT_STORAGE = Path(os.getenv("CAREERAI_REPORT_STORAGE", PROJECT_ROOT / "data" / "private_reports"))
MAX_RESUME_BYTES = 10 * 1024 * 1024


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	full_name TEXT NOT NULL,
	email TEXT NOT NULL UNIQUE COLLATE NOCASE,
	password_hash TEXT NOT NULL,
	role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
	is_active INTEGER NOT NULL DEFAULT 1,
	email_verified INTEGER NOT NULL DEFAULT 0,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS password_reset_tokens (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	token_hash TEXT NOT NULL UNIQUE,
	expires_at TEXT NOT NULL,
	used_at TEXT,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS email_verification_tokens (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	token_hash TEXT NOT NULL UNIQUE,
	expires_at TEXT NOT NULL,
	used_at TEXT,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS login_audit (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
	email TEXT NOT NULL,
	was_successful INTEGER NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reset_token_hash ON password_reset_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_verification_token_hash ON email_verification_tokens(token_hash);
"""

RESUME_SCHEMA = """
CREATE TABLE IF NOT EXISTS resumes (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	original_filename TEXT NOT NULL,
	storage_key TEXT NOT NULL UNIQUE,
	sha256 TEXT NOT NULL,
	file_size INTEGER NOT NULL,
	is_current INTEGER NOT NULL DEFAULT 1,
	uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_resumes_user_current ON resumes(user_id, is_current, uploaded_at);
CREATE TABLE IF NOT EXISTS resume_analyses (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	resume_id INTEGER NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
	extracted_data_json TEXT NOT NULL,
	section_data_json TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_user_date ON resume_analyses(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ats_scores (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	analysis_id INTEGER NOT NULL UNIQUE REFERENCES resume_analyses(id) ON DELETE CASCADE,
	overall INTEGER NOT NULL CHECK (overall BETWEEN 0 AND 100),
	skills INTEGER NOT NULL CHECK (skills BETWEEN 0 AND 100),
	keywords INTEGER NOT NULL CHECK (keywords BETWEEN 0 AND 100),
	structure INTEGER NOT NULL CHECK (structure BETWEEN 0 AND 100),
	education INTEGER NOT NULL CHECK (education BETWEEN 0 AND 100),
	experience INTEGER NOT NULL CHECK (experience BETWEEN 0 AND 100),
	projects INTEGER NOT NULL CHECK (projects BETWEEN 0 AND 100),
	completeness INTEGER NOT NULL CHECK (completeness BETWEEN 0 AND 100),
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS extracted_skills (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	analysis_id INTEGER NOT NULL REFERENCES resume_analyses(id) ON DELETE CASCADE,
	skill TEXT NOT NULL,
	category TEXT NOT NULL,
	source TEXT NOT NULL CHECK (source IN ('resume', 'profile', 'recommended', 'missing')),
	UNIQUE(analysis_id, skill, source)
);
CREATE INDEX IF NOT EXISTS idx_extracted_skills_user ON extracted_skills(user_id, source, skill);
CREATE TABLE IF NOT EXISTS resume_suggestions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	analysis_id INTEGER NOT NULL REFERENCES resume_analyses(id) ON DELETE CASCADE,
	priority TEXT NOT NULL,
	suggestion TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_profiles (
	user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
	career_goal TEXT NOT NULL DEFAULT '',
	education TEXT NOT NULL DEFAULT '',
	experience_summary TEXT NOT NULL DEFAULT '',
	skills_json TEXT NOT NULL DEFAULT '[]',
	interests_json TEXT NOT NULL DEFAULT '[]',
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS generated_reports (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	analysis_id INTEGER NOT NULL REFERENCES resume_analyses(id) ON DELETE CASCADE,
	storage_key TEXT NOT NULL UNIQUE,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CAREER_GUIDANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS career_roles (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	career_name TEXT NOT NULL UNIQUE,
	description TEXT NOT NULL,
	overview TEXT NOT NULL,
	learning_direction TEXT NOT NULL,
	future_roles_json TEXT NOT NULL DEFAULT '[]',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS career_required_skills (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	career_id INTEGER NOT NULL REFERENCES career_roles(id) ON DELETE CASCADE,
	skill TEXT NOT NULL,
	category TEXT NOT NULL,
	is_optional INTEGER NOT NULL DEFAULT 0,
	UNIQUE(career_id, skill, category)
);
CREATE TABLE IF NOT EXISTS career_roadmaps (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	career_id INTEGER NOT NULL REFERENCES career_roles(id) ON DELETE CASCADE,
	stage_name TEXT NOT NULL,
	topics_json TEXT NOT NULL,
	skills_json TEXT NOT NULL,
	projects_json TEXT NOT NULL,
	outcome TEXT NOT NULL,
	progress_label TEXT NOT NULL,
	UNIQUE(career_id, stage_name)
);
CREATE TABLE IF NOT EXISTS career_learning_paths (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	career_id INTEGER NOT NULL REFERENCES career_roles(id) ON DELETE CASCADE,
	recommended_topic TEXT NOT NULL,
	reason TEXT NOT NULL,
	priority TEXT NOT NULL,
	estimated_stage TEXT NOT NULL,
	related_skills_json TEXT NOT NULL,
	UNIQUE(career_id, recommended_topic)
);
CREATE TABLE IF NOT EXISTS user_career_recommendations (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	career_name TEXT NOT NULL,
	recommendation_score INTEGER NOT NULL,
	why_suitable TEXT NOT NULL,
	matching_skills_json TEXT NOT NULL,
	missing_skills_json TEXT NOT NULL,
	required_skills_json TEXT NOT NULL,
	learning_direction TEXT NOT NULL,
	overview TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_skill_gaps (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	career_name TEXT NOT NULL,
	matching_skills_json TEXT NOT NULL,
	missing_skills_json TEXT NOT NULL,
	partially_matched_json TEXT NOT NULL,
	skill_gap_percentage INTEGER NOT NULL,
	priority_missing_json TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_career_progress (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	career_name TEXT NOT NULL,
	roadmap_progress INTEGER NOT NULL DEFAULT 0,
	learning_progress INTEGER NOT NULL DEFAULT 0,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, career_name)
);
"""

LEARNING_HUB_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_categories (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	slug TEXT NOT NULL UNIQUE,
	name TEXT NOT NULL,
	description TEXT NOT NULL,
	icon TEXT NOT NULL DEFAULT ':material/school:',
	difficulty TEXT NOT NULL DEFAULT 'Beginner',
	duration TEXT NOT NULL DEFAULT '2 weeks'
);
CREATE TABLE IF NOT EXISTS learning_courses (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	category_id INTEGER NOT NULL REFERENCES learning_categories(id) ON DELETE CASCADE,
	course_name TEXT NOT NULL,
	description TEXT NOT NULL,
	difficulty TEXT NOT NULL DEFAULT 'Beginner',
	duration TEXT NOT NULL DEFAULT '2 weeks',
	icon TEXT NOT NULL DEFAULT ':material/school:',
	tags_json TEXT NOT NULL DEFAULT '[]',
	UNIQUE(category_id, course_name)
);
CREATE TABLE IF NOT EXISTS learning_topics (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	course_id INTEGER NOT NULL REFERENCES learning_courses(id) ON DELETE CASCADE,
	topic_name TEXT NOT NULL,
	explanation TEXT NOT NULL,
	notes_json TEXT NOT NULL DEFAULT '[]',
	examples_json TEXT NOT NULL DEFAULT '[]',
	questions_json TEXT NOT NULL DEFAULT '[]',
	status TEXT NOT NULL DEFAULT 'Not started',
	UNIQUE(course_id, topic_name)
);
CREATE TABLE IF NOT EXISTS learning_lessons (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	topic_id INTEGER NOT NULL REFERENCES learning_topics(id) ON DELETE CASCADE,
	lesson_title TEXT NOT NULL,
	lesson_content TEXT NOT NULL,
	UNIQUE(topic_id, lesson_title)
);
CREATE TABLE IF NOT EXISTS learning_projects (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	course_id INTEGER NOT NULL REFERENCES learning_courses(id) ON DELETE CASCADE,
	project_title TEXT NOT NULL,
	difficulty TEXT NOT NULL,
	technologies_json TEXT NOT NULL DEFAULT '[]',
	objective TEXT NOT NULL,
	required_skills_json TEXT NOT NULL DEFAULT '[]',
	project_steps_json TEXT NOT NULL DEFAULT '[]',
	expected_outcome TEXT NOT NULL,
	UNIQUE(course_id, project_title)
);
CREATE TABLE IF NOT EXISTS learning_user_course_progress (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	course_id INTEGER NOT NULL REFERENCES learning_courses(id) ON DELETE CASCADE,
	progress_percent INTEGER NOT NULL DEFAULT 0,
	last_topic TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'Not started',
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, course_id)
);
CREATE TABLE IF NOT EXISTS learning_user_topic_progress (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	topic_id INTEGER NOT NULL REFERENCES learning_topics(id) ON DELETE CASCADE,
	progress_percent INTEGER NOT NULL DEFAULT 0,
	status TEXT NOT NULL DEFAULT 'Not started',
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, topic_id)
);
CREATE TABLE IF NOT EXISTS learning_user_project_progress (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	project_id INTEGER NOT NULL REFERENCES learning_projects(id) ON DELETE CASCADE,
	progress_percent INTEGER NOT NULL DEFAULT 0,
	status TEXT NOT NULL DEFAULT 'Not started',
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, project_id)
);
CREATE TABLE IF NOT EXISTS learning_activity (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	course_name TEXT NOT NULL,
	topic_name TEXT NOT NULL,
	progress_percent INTEGER NOT NULL DEFAULT 0,
	activity_type TEXT NOT NULL DEFAULT 'learning',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

RESOURCES_SCHEMA = """
CREATE TABLE IF NOT EXISTS resource_categories (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	category_name TEXT NOT NULL UNIQUE,
	description TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS resource_types (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	type_name TEXT NOT NULL UNIQUE,
	description TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS resources (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	type_id INTEGER NOT NULL REFERENCES resource_types(id) ON DELETE CASCADE,
	category_id INTEGER NOT NULL REFERENCES resource_categories(id) ON DELETE CASCADE,
	title TEXT NOT NULL,
	description TEXT NOT NULL,
	technology TEXT NOT NULL,
	topic TEXT NOT NULL,
	difficulty TEXT NOT NULL,
	url TEXT,
	file_path TEXT,
	source TEXT NOT NULL DEFAULT 'internal',
	active INTEGER NOT NULL DEFAULT 1,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS resource_metadata (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
	meta_key TEXT NOT NULL,
	meta_value TEXT NOT NULL,
	UNIQUE(resource_id, meta_key)
);
CREATE TABLE IF NOT EXISTS resource_bookmarks (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, resource_id)
);
CREATE TABLE IF NOT EXISTS resource_progress (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
	status TEXT NOT NULL DEFAULT 'Not Started',
	progress_percent INTEGER NOT NULL DEFAULT 0,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	UNIQUE(user_id, resource_id)
);
"""


AI_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_conversations (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	title TEXT NOT NULL DEFAULT 'AI Assistant',
	mode TEXT NOT NULL DEFAULT 'Chatbot',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ai_messages (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	conversation_id INTEGER NOT NULL REFERENCES ai_conversations(id) ON DELETE CASCADE,
	role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
	content TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS study_plans (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	title TEXT NOT NULL,
	target_career TEXT NOT NULL,
	daily_minutes INTEGER NOT NULL DEFAULT 60,
	weekly_days INTEGER NOT NULL DEFAULT 5,
	target_weeks INTEGER NOT NULL DEFAULT 4,
	status TEXT NOT NULL DEFAULT 'Active',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS study_plan_tasks (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	plan_id INTEGER NOT NULL REFERENCES study_plans(id) ON DELETE CASCADE,
	day_label TEXT NOT NULL,
	topic TEXT NOT NULL,
	resource TEXT NOT NULL,
	practice TEXT NOT NULL,
	task TEXT NOT NULL,
	estimated_minutes INTEGER NOT NULL DEFAULT 60,
	completion_status TEXT NOT NULL DEFAULT 'Pending',
	sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ai_suggestions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	category TEXT NOT NULL,
	title TEXT NOT NULL,
	details TEXT NOT NULL,
	action_label TEXT NOT NULL,
	resource_url TEXT,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ai_mentor_sessions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	focus TEXT NOT NULL,
	summary TEXT NOT NULL,
	recommendation TEXT NOT NULL,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ai_conversations_user ON ai_conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_messages_conversation ON ai_messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_user ON ai_suggestions(user_id, created_at DESC);
"""

JOBS_REPORTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS activity_history (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	activity TEXT NOT NULL,
	module TEXT NOT NULL,
	status TEXT NOT NULL DEFAULT 'Completed',
	description TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_activity_history_user_date
	ON activity_history(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS practice_attempts (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	kind TEXT NOT NULL,
	subject TEXT NOT NULL,
	score REAL,
	total REAL,
	status TEXT NOT NULL DEFAULT 'Completed',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_practice_attempts_user_date
	ON practice_attempts(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS email_report_logs (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	report_type TEXT NOT NULL,
	recipient TEXT NOT NULL,
	status TEXT NOT NULL,
	message TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_email_report_logs_user_date
	ON email_report_logs(user_id, created_at DESC);
"""

ADMIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_activity_logs (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	admin_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	action TEXT NOT NULL,
	module TEXT NOT NULL,
	target_record TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_admin_activity_date ON admin_activity_logs(created_at DESC);
CREATE TABLE IF NOT EXISTS user_feedback (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	feedback_type TEXT NOT NULL,
	message TEXT NOT NULL,
	rating INTEGER CHECK (rating BETWEEN 1 AND 5),
	status TEXT NOT NULL DEFAULT 'New' CHECK (status IN ('New', 'Reviewed', 'In Progress', 'Resolved')),
	admin_note TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_feedback_date ON user_feedback(created_at DESC);
CREATE TABLE IF NOT EXISTS ai_model_registry (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	service_key TEXT NOT NULL UNIQUE,
	service_name TEXT NOT NULL,
	purpose TEXT NOT NULL,
	technology TEXT NOT NULL,
	version TEXT NOT NULL,
	model_type TEXT NOT NULL,
	description TEXT NOT NULL,
	enabled INTEGER NOT NULL DEFAULT 1,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS admin_report_logs (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	admin_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
	report_type TEXT NOT NULL,
	start_date TEXT,
	end_date TEXT,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def initialize_database() -> None:
	DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
	with connection() as database:
		database.executescript(SCHEMA)
		database.executescript(RESUME_SCHEMA)
		database.executescript(CAREER_GUIDANCE_SCHEMA)
		database.executescript(LEARNING_HUB_SCHEMA)
		database.executescript(RESOURCES_SCHEMA)
		database.executescript(AI_SCHEMA)
		database.executescript(JOBS_REPORTS_SCHEMA)
		database.executescript(ADMIN_SCHEMA)
		course_columns = {row["name"] for row in database.execute("PRAGMA table_info(learning_courses)")}
		if "active" not in course_columns:
			database.execute("ALTER TABLE learning_courses ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
		seed_career_dataset()
		seed_learning_hub_data()
		seed_resources_data()
		seed_ai_model_registry(database)


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
	"""Short-lived connection boundary; replace this implementation for MySQL later."""
	DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
	database = sqlite3.connect(DATABASE_PATH)
	database.row_factory = sqlite3.Row
	database.execute("PRAGMA foreign_keys = ON")
	try:
		yield database
		database.commit()
	except Exception:
		database.rollback()
		raise
	finally:
		database.close()


def record_activity(user_id: int, activity: str, module: str, description: str, status: str = "Completed") -> None:
	with connection() as database:
		database.execute(
			"INSERT INTO activity_history (user_id, activity, module, status, description) VALUES (?, ?, ?, ?, ?)",
			(user_id, activity[:120], module[:80], status[:40], description[:500]),
		)


def seed_ai_model_registry(database: sqlite3.Connection) -> None:
	services = [
		("ats_scoring", "Resume ATS scoring", "Score resume structure, skills, keywords, and completeness.", "Python rule-based scoring", "1.0", "Heuristic", "Deterministic scoring; this is not a trained ML model."),
		("skill_extraction", "Skill extraction", "Identify skills and resume sections.", "spaCy-independent pattern matching", "1.0", "NLP rules", "Uses the configured skill catalog and text patterns."),
		("career_recommendations", "Career recommendations", "Rank career paths against profile and resume content.", "scikit-learn TF-IDF + cosine similarity", "1.1", "Unsupervised content-based ML", "Ranks role descriptions by text similarity to saved profile and resume data; it is not trained on labeled user outcomes."),
		("skill_gap_analysis", "Skill gap analysis", "Compare saved skills with career requirements.", "Python rule-based matching", "1.0", "Analysis rules", "Compares the user's skills with the configured career dataset."),
		("job_matching", "Job matching", "Compare user profiles with the curated sample role catalog.", "Python rule-based matching", "1.0", "Matching rules", "Uses sample roles; it is not connected to a live job board."),
		("ai_assistant", "AI Assistant", "Answer career and learning questions.", get_ai_provider() if "get_ai_provider" in globals() else "Local / configured LLM", get_ai_model() if "get_ai_model" in globals() else "offline-local", "Local / LLM service", "Uses the existing local response or configured external provider."),
		("mock_interview", "AI Mock Interview", "Provide keyword-based mock interview feedback.", "Python keyword matching", "1.0", "Heuristic", "Feedback is a keyword heuristic, not a trained interview model."),
	]
	for service in services:
		database.execute(
			"""INSERT OR IGNORE INTO ai_model_registry
			   (service_key, service_name, purpose, technology, version, model_type, description)
			   VALUES (?, ?, ?, ?, ?, ?, ?)""",
			service,
		)
	database.execute(
		"""UPDATE ai_model_registry
		   SET purpose = ?, technology = ?, version = ?, model_type = ?, description = ?,
		       updated_at = CURRENT_TIMESTAMP
		   WHERE service_key = 'career_recommendations'
		     AND technology = 'Python rule-based matching'""",
		(
			"Rank career paths against profile and resume content.",
			"scikit-learn TF-IDF + cosine similarity",
			"1.1",
			"Unsupervised content-based ML",
			"Ranks role descriptions by text similarity to saved profile and resume data; it is not trained on labeled user outcomes.",
		),
	)


def record_practice_attempt(
	user_id: int,
	kind: str,
	subject: str,
	score: float | None = None,
	total: float | None = None,
	status: str = "Completed",
) -> None:
	with connection() as database:
		database.execute(
			"INSERT INTO practice_attempts (user_id, kind, subject, score, total, status) VALUES (?, ?, ?, ?, ?, ?)",
			(user_id, kind[:100], subject[:160], score, total, status[:40]),
		)


# -----------------------------------------------------------------------------
# Security and validation
# -----------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_email(email: str) -> str:
	return email.strip().lower()


def validate_email(email: str) -> bool:
	return bool(EMAIL_PATTERN.fullmatch(normalize_email(email)))


def password_errors(password: str) -> list[str]:
	errors = []
	if len(password) < 8:
		errors.append("at least 8 characters")
	if not re.search(r"[A-Z]", password):
		errors.append("an uppercase letter")
	if not re.search(r"[a-z]", password):
		errors.append("a lowercase letter")
	if not re.search(r"\d", password):
		errors.append("a number")
	if not re.search(r"[^A-Za-z0-9]", password):
		errors.append("a special character")
	return errors


def hash_password(password: str) -> str:
	return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
	try:
		return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
	except (ValueError, TypeError):
		return False


def new_token() -> str:
	return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
	return hashlib.sha256(token.encode("utf-8")).hexdigest()


def expiry(**duration: int) -> str:
	return (datetime.now(timezone.utc) + timedelta(**duration)).isoformat()


# -----------------------------------------------------------------------------
# Database repository
# -----------------------------------------------------------------------------

def get_user(email: str) -> dict[str, Any] | None:
	with connection() as database:
		row = database.execute("SELECT * FROM users WHERE email = ?", (normalize_email(email),)).fetchone()
	return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
	with connection() as database:
		row = database.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
	return dict(row) if row else None


def insert_user(full_name: str, email: str, password: str) -> dict[str, Any]:
	with connection() as database:
		cursor = database.execute(
			"INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
			(full_name.strip(), normalize_email(email), hash_password(password)),
		)
		user_id = cursor.lastrowid
	return get_user_by_id(user_id)


def create_token(table: str, user_id: int, raw_token: str, expires_at: str) -> None:
	if table not in {"password_reset_tokens", "email_verification_tokens"}:
		raise ValueError("Unsupported token table")
	with connection() as database:
		database.execute(f"UPDATE {table} SET used_at = CURRENT_TIMESTAMP WHERE user_id = ? AND used_at IS NULL", (user_id,))
		database.execute(
			f"INSERT INTO {table} (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
			(user_id, token_hash(raw_token), expires_at),
		)


def valid_token(table: str, raw_token: str) -> dict[str, Any] | None:
	if table not in {"password_reset_tokens", "email_verification_tokens"}:
		raise ValueError("Unsupported token table")
	with connection() as database:
		row = database.execute(
			f"SELECT * FROM {table} WHERE token_hash = ? AND used_at IS NULL",
			(token_hash(raw_token.strip()),),
		).fetchone()
	if not row:
		return None
	expires_at = datetime.fromisoformat(row["expires_at"])
	if expires_at <= datetime.now(timezone.utc):
		return None
	return dict(row)


def consume_token(table: str, token_id: int) -> None:
	with connection() as database:
		database.execute(f"UPDATE {table} SET used_at = CURRENT_TIMESTAMP WHERE id = ?", (token_id,))


# -----------------------------------------------------------------------------
# Authentication service
# -----------------------------------------------------------------------------

def register_user(full_name: str, email: str, password: str, confirmation: str) -> tuple[bool, str, str | None]:
	if len(full_name.strip()) < 2:
		return False, "Enter your full name.", None
	if not validate_email(email):
		return False, "Enter a valid email address.", None
	errors = password_errors(password)
	if errors:
		return False, "Your password needs " + ", ".join(errors) + ".", None
	if password != confirmation:
		return False, "Passwords do not match.", None
	try:
		user = insert_user(full_name, email, password)
	except sqlite3.IntegrityError:
		return False, "An account with this email already exists.", None
	verification_token = new_token()
	create_token("email_verification_tokens", user["id"], verification_token, expiry(hours=VERIFICATION_TOKEN_HOURS))
	return True, "Account created. Verify your email to unlock all features.", verification_token


def login_user(email: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
	email = normalize_email(email)
	user = get_user(email)
	valid = bool(user and user["is_active"] and verify_password(password, user["password_hash"]))
	with connection() as database:
		database.execute(
			"INSERT INTO login_audit (user_id, email, was_successful) VALUES (?, ?, ?)",
			(user["id"] if user else None, email, int(valid)),
		)
	if not valid:
		return False, "Email or password is incorrect.", None
	if email.casefold() == ADMIN_EMAIL and user["role"] != "admin":
		with connection() as database:
			database.execute(
				"UPDATE users SET role = 'admin', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
				(user["id"],),
			)
		user = get_user_by_id(int(user["id"]))
		log_admin_action(int(user["id"]), "Granted configured administrator role", "User Management", f"user_id={user['id']}")
	if user["role"] == "admin":
		log_admin_action(int(user["id"]), "Admin logged in", "Authentication", f"user_id={user['id']}")
	return True, "Welcome back.", user


def require_admin(admin_user_id: int) -> dict[str, Any]:
	"""Reload authorization from the database; session role data is never trusted."""
	with connection() as database:
		row = database.execute(
			"SELECT id, full_name, email, role, is_active FROM users WHERE id = ?",
			(admin_user_id,),
		).fetchone()
	if (
		not row
		or row["role"] != "admin"
		or not row["is_active"]
		or row["email"].casefold() != ADMIN_EMAIL
	):
		raise PermissionError("Access Denied — Administrator privileges required.")
	return dict(row)


def log_admin_action(admin_user_id: int, action: str, module: str, target_record: str = "") -> None:
	require_admin(admin_user_id)
	with connection() as database:
		database.execute(
			"INSERT INTO admin_activity_logs (admin_user_id, action, module, target_record) VALUES (?, ?, ?, ?)",
			(admin_user_id, action[:160], module[:80], target_record[:160]),
		)


def admin_user_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		rows = database.execute(
			"""SELECT u.id, u.full_name, u.email, u.role, u.is_active, u.created_at,
				(SELECT MAX(created_at) FROM activity_history WHERE user_id = u.id) AS last_activity,
				(SELECT MAX(created_at) FROM login_audit WHERE user_id = u.id AND was_successful = 1) AS last_login,
				(SELECT overall FROM ats_scores WHERE user_id = u.id ORDER BY created_at DESC, id DESC LIMIT 1) AS resume_score
			   FROM users u ORDER BY u.created_at DESC, u.id DESC"""
		).fetchall()
		user_data = [dict(row) for row in rows]
	for item in user_data:
		profile = get_user_profile(int(item["id"]))
		item["profile_completion"] = profile_completion(item, profile)
		item["last_activity"] = max(
			(value for value in (item["last_activity"], item["last_login"]) if value),
			default=None,
		)
		item["career_readiness"] = get_jobs_report_progress(int(item["id"]))["career_readiness"]
	return user_data


def admin_update_user(admin_user_id: int, target_user_id: int, role: str, is_active: bool) -> None:
	require_admin(admin_user_id)
	if role not in {"user", "admin"}:
		raise ValueError("Choose a supported account role.")
	with connection() as database:
		target = database.execute("SELECT id, email, role, is_active FROM users WHERE id = ?", (target_user_id,)).fetchone()
		if not target:
			raise ValueError("The selected account no longer exists.")
		if int(target["id"]) == admin_user_id:
			raise ValueError("You cannot change your own role or account status.")
		if target["email"].casefold() == ADMIN_EMAIL and (role != "admin" or not is_active):
			raise ValueError("The configured administrator account cannot be demoted or deactivated here.")
		if target["role"] == "admin" and (role != "admin" or not is_active):
			other_admin = database.execute(
				"SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
				(target_user_id,),
			).fetchone()[0]
			if not other_admin:
				raise ValueError("Keep at least one active administrator account.")
		database.execute(
			"UPDATE users SET role = ?, is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
			(role, int(is_active), target_user_id),
		)
	log_admin_action(admin_user_id, "Updated account role/status", "User Management", f"user_id={target_user_id}")


def admin_course_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		return [dict(row) for row in database.execute(
			"""SELECT c.id, c.course_name, c.description, c.difficulty, c.duration, c.icon,
				c.tags_json, c.active, cat.id AS category_id, cat.name AS category_name,
				(SELECT COUNT(*) FROM learning_topics WHERE course_id = c.id) AS topic_count,
				(SELECT COUNT(*) FROM learning_projects WHERE course_id = c.id) AS project_count
			   FROM learning_courses c JOIN learning_categories cat ON cat.id = c.category_id
			   ORDER BY cat.name, c.course_name"""
		).fetchall()]


def admin_save_course(
	admin_user_id: int,
	course_id: int | None,
	category_name: str,
	course_name: str,
	description: str,
	difficulty: str,
	duration: str,
	tags: list[str],
) -> None:
	require_admin(admin_user_id)
	category_name = category_name.strip()
	course_name = course_name.strip()
	description = description.strip()
	if not category_name or not course_name or len(course_name) > 120 or not description:
		raise ValueError("Enter a category, a course name (up to 120 characters), and a description.")
	if len(description) > 4000:
		raise ValueError("Course descriptions must be 4,000 characters or fewer.")
	slug = re.sub(r"[^a-z0-9]+", "-", category_name.casefold()).strip("-")[:80]
	if not slug:
		raise ValueError("Enter a valid course category.")
	with connection() as database:
		database.execute(
			"INSERT OR IGNORE INTO learning_categories (slug, name, description) VALUES (?, ?, ?)",
			(slug, category_name[:80], f"Learning content for {category_name[:80]}."),
		)
		category = database.execute("SELECT id FROM learning_categories WHERE slug = ?", (slug,)).fetchone()
		if not category:
			raise ValueError("A category with a conflicting identifier already exists.")
		if course_id is None:
			database.execute(
				"""INSERT INTO learning_courses
				   (category_id, course_name, description, difficulty, duration, tags_json)
				   VALUES (?, ?, ?, ?, ?, ?)""",
				(int(category["id"]), course_name, description, difficulty, duration, json.dumps(tags)),
			)
			action = "Created course"
		else:
			database.execute(
				"""UPDATE learning_courses SET category_id = ?, course_name = ?, description = ?,
				   difficulty = ?, duration = ?, tags_json = ? WHERE id = ?""",
				(int(category["id"]), course_name, description, difficulty, duration, json.dumps(tags), course_id),
			)
			action = "Updated course"
	log_admin_action(admin_user_id, action, "Course Management", f"course_id={course_id or course_name}")


def admin_set_course_active(admin_user_id: int, course_id: int, active: bool) -> None:
	require_admin(admin_user_id)
	with connection() as database:
		result = database.execute("UPDATE learning_courses SET active = ? WHERE id = ?", (int(active), course_id))
		if not result.rowcount:
			raise ValueError("The selected course no longer exists.")
	log_admin_action(admin_user_id, "Changed course visibility", "Course Management", f"course_id={course_id}")


def admin_save_topic(
	admin_user_id: int,
	course_id: int,
	topic_id: int | None,
	title: str,
	explanation: str,
	notes: list[str],
	examples: list[str],
	questions: list[str],
) -> None:
	require_admin(admin_user_id)
	title, explanation = title.strip(), explanation.strip()
	if not title or len(title) > 160 or not explanation:
		raise ValueError("Enter a topic title (up to 160 characters) and an explanation.")
	with connection() as database:
		if topic_id is None:
			database.execute(
				"""INSERT INTO learning_topics
				   (course_id, topic_name, explanation, notes_json, examples_json, questions_json)
				   VALUES (?, ?, ?, ?, ?, ?)""",
				(course_id, title, explanation, json.dumps(notes), json.dumps(examples), json.dumps(questions)),
			)
			target = title
			action = "Created course topic"
		else:
			result = database.execute(
				"""UPDATE learning_topics SET topic_name = ?, explanation = ?, notes_json = ?,
				   examples_json = ?, questions_json = ? WHERE id = ? AND course_id = ?""",
				(title, explanation, json.dumps(notes), json.dumps(examples), json.dumps(questions), topic_id, course_id),
			)
			if not result.rowcount:
				raise ValueError("The selected topic no longer exists.")
			target = f"topic_id={topic_id}"
			action = "Updated course topic"
	log_admin_action(admin_user_id, action, "Course Management", target)


def admin_save_lesson(admin_user_id: int, topic_id: int, lesson_id: int | None, title: str, content: str) -> None:
	require_admin(admin_user_id)
	title, content = title.strip(), content.strip()
	if not title or len(title) > 160 or not content:
		raise ValueError("Enter a lesson title (up to 160 characters) and lesson content.")
	with connection() as database:
		if lesson_id is None:
			database.execute(
				"INSERT INTO learning_lessons (topic_id, lesson_title, lesson_content) VALUES (?, ?, ?)",
				(topic_id, title, content),
			)
			target = title
			action = "Created lesson"
		else:
			result = database.execute(
				"UPDATE learning_lessons SET lesson_title = ?, lesson_content = ? WHERE id = ? AND topic_id = ?",
				(title, content, lesson_id, topic_id),
			)
			if not result.rowcount:
				raise ValueError("The selected lesson no longer exists.")
			target = f"lesson_id={lesson_id}"
			action = "Updated lesson"
	log_admin_action(admin_user_id, action, "Course Management", target)


def admin_save_project(
	admin_user_id: int,
	course_id: int,
	project_id: int | None,
	title: str,
	difficulty: str,
	technologies: list[str],
	objective: str,
	required_skills: list[str],
	steps: list[str],
	outcome: str,
) -> None:
	require_admin(admin_user_id)
	title, objective, outcome = title.strip(), objective.strip(), outcome.strip()
	if not title or not objective or not outcome:
		raise ValueError("Enter a project title, objective, and expected outcome.")
	values = (title, difficulty, json.dumps(technologies), objective, json.dumps(required_skills), json.dumps(steps), outcome)
	with connection() as database:
		if project_id is None:
			database.execute(
				"""INSERT INTO learning_projects
				   (course_id, project_title, difficulty, technologies_json, objective,
				    required_skills_json, project_steps_json, expected_outcome)
				   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
				(course_id, *values),
			)
			target = title
			action = "Created course project"
		else:
			result = database.execute(
				"""UPDATE learning_projects SET project_title = ?, difficulty = ?, technologies_json = ?,
				   objective = ?, required_skills_json = ?, project_steps_json = ?, expected_outcome = ?
				   WHERE id = ? AND course_id = ?""",
				(*values, project_id, course_id),
			)
			if not result.rowcount:
				raise ValueError("The selected project no longer exists.")
			target = f"project_id={project_id}"
			action = "Updated course project"
	log_admin_action(admin_user_id, action, "Course Management", target)


def admin_resource_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		return [dict(row) for row in database.execute(
			"""SELECT r.id, r.title, r.description, r.technology, r.topic, r.difficulty,
				r.url, r.file_path, r.source, r.active, r.created_at, r.updated_at,
				rt.id AS type_id, rt.type_name, rc.id AS category_id, rc.category_name
			   FROM resources r JOIN resource_types rt ON rt.id = r.type_id
			   JOIN resource_categories rc ON rc.id = r.category_id
			   ORDER BY r.updated_at DESC, r.id DESC"""
		).fetchall()]


def admin_save_resource(
	admin_user_id: int,
	resource_id: int | None,
	title: str,
	description: str,
	category_name: str,
	resource_type: str,
	technology: str,
	topic: str,
	difficulty: str,
	url: str,
) -> None:
	require_admin(admin_user_id)
	title, description, url = title.strip(), description.strip(), url.strip()
	if not title or not description or not category_name.strip() or not resource_type.strip():
		raise ValueError("Enter a title, description, category, and resource type.")
	if url and not re.fullmatch(r"https://[^\s]+", url, re.IGNORECASE):
		raise ValueError("Resource URLs must use HTTPS.")
	with connection() as database:
		database.execute(
			"INSERT OR IGNORE INTO resource_categories (category_name, description) VALUES (?, ?)",
			(category_name.strip()[:80], f"Resources in {category_name.strip()[:80]}."),
		)
		database.execute(
			"INSERT OR IGNORE INTO resource_types (type_name, description) VALUES (?, ?)",
			(resource_type.strip()[:80], f"{resource_type.strip()[:80]} resources."),
		)
		category = database.execute("SELECT id FROM resource_categories WHERE category_name = ?", (category_name.strip(),)).fetchone()
		type_row = database.execute("SELECT id FROM resource_types WHERE type_name = ?", (resource_type.strip(),)).fetchone()
		if not category or not type_row:
			raise ValueError("Choose a valid resource category and resource type.")
		values = (
			int(type_row["id"]), int(category["id"]), title[:160], description[:4000],
			technology.strip()[:160], topic.strip()[:160], difficulty, url or None,
		)
		if resource_id is None:
			database.execute(
				"""INSERT INTO resources
				   (type_id, category_id, title, description, technology, topic, difficulty, url, source, active)
				   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'admin', 1)""",
				values,
			)
			target = title
			action = "Created resource"
		else:
			result = database.execute(
				"""UPDATE resources SET type_id = ?, category_id = ?, title = ?, description = ?,
				   technology = ?, topic = ?, difficulty = ?, url = ?, updated_at = CURRENT_TIMESTAMP
				   WHERE id = ?""",
				(*values, resource_id),
			)
			if not result.rowcount:
				raise ValueError("The selected resource no longer exists.")
			target = f"resource_id={resource_id}"
			action = "Updated resource"
	log_admin_action(admin_user_id, action, "Resource Management", target)


def admin_set_resource_active(admin_user_id: int, resource_id: int, active: bool) -> None:
	require_admin(admin_user_id)
	with connection() as database:
		result = database.execute(
			"UPDATE resources SET active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
			(int(active), resource_id),
		)
		if not result.rowcount:
			raise ValueError("The selected resource no longer exists.")
	log_admin_action(admin_user_id, "Changed resource visibility", "Resource Management", f"resource_id={resource_id}")


def admin_delete_resource(admin_user_id: int, resource_id: int) -> None:
	require_admin(admin_user_id)
	with connection() as database:
		references = database.execute(
			"""SELECT (SELECT COUNT(*) FROM resource_bookmarks WHERE resource_id = ?) +
					  (SELECT COUNT(*) FROM resource_progress WHERE resource_id = ?) AS total""",
			(resource_id, resource_id),
		).fetchone()["total"]
		if references:
			raise ValueError("This resource has user bookmarks or progress. Deactivate it instead of deleting it.")
		result = database.execute("DELETE FROM resources WHERE id = ?", (resource_id,))
		if not result.rowcount:
			raise ValueError("The selected resource no longer exists.")
	log_admin_action(admin_user_id, "Deleted unused resource", "Resource Management", f"resource_id={resource_id}")


def admin_feedback_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		return [dict(row) for row in database.execute(
			"""SELECT f.id, f.feedback_type, f.message, f.rating, f.status, f.admin_note,
				f.created_at, u.full_name AS user_name
			   FROM user_feedback f JOIN users u ON u.id = f.user_id
			   ORDER BY f.created_at DESC, f.id DESC"""
		).fetchall()]


def admin_add_feedback(
	admin_user_id: int,
	user_id: int,
	feedback_type: str,
	message: str,
	rating: int | None,
) -> None:
	require_admin(admin_user_id)
	feedback_type, message = feedback_type.strip(), message.strip()
	if not feedback_type or len(feedback_type) > 80 or not message or len(message) > 4000:
		raise ValueError("Enter a feedback type and a message of up to 4,000 characters.")
	if rating is not None and rating not in range(1, 6):
		raise ValueError("Rating must be between 1 and 5.")
	with connection() as database:
		if not database.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
			raise ValueError("The selected account no longer exists.")
		database.execute(
			"INSERT INTO user_feedback (user_id, feedback_type, message, rating) VALUES (?, ?, ?, ?)",
			(user_id, feedback_type, message, rating),
		)
	log_admin_action(admin_user_id, "Recorded received user feedback", "Feedback Management", f"user_id={user_id}")


def admin_update_feedback(admin_user_id: int, feedback_id: int, status: str, admin_note: str) -> None:
	require_admin(admin_user_id)
	if status not in {"New", "Reviewed", "In Progress", "Resolved"}:
		raise ValueError("Choose a valid feedback status.")
	with connection() as database:
		result = database.execute(
			"UPDATE user_feedback SET status = ?, admin_note = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
			(status, admin_note[:2000], feedback_id),
		)
		if not result.rowcount:
			raise ValueError("The selected feedback no longer exists.")
	log_admin_action(admin_user_id, "Updated feedback status/notes", "Feedback Management", f"feedback_id={feedback_id}")


def admin_model_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		return [dict(row) for row in database.execute(
			"SELECT * FROM ai_model_registry ORDER BY service_name"
		).fetchall()]


def admin_update_model(admin_user_id: int, service_id: int, version: str, technology: str, description: str) -> None:
	require_admin(admin_user_id)
	if not version.strip() or not technology.strip() or not description.strip():
		raise ValueError("Version, technology, and description are required.")
	with connection() as database:
		result = database.execute(
			"""UPDATE ai_model_registry SET version = ?, technology = ?, description = ?,
			   updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
			(version.strip()[:100], technology.strip()[:200], description.strip()[:1000], service_id),
		)
		if not result.rowcount:
			raise ValueError("The selected AI service no longer exists.")
	log_admin_action(admin_user_id, "Updated AI service metadata", "AI Model Management", f"service_id={service_id}")


def admin_set_external_ai_enabled(admin_user_id: int, enabled: bool) -> None:
	require_admin(admin_user_id)
	with connection() as database:
		database.execute(
			"UPDATE ai_model_registry SET enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE service_key = 'ai_assistant'",
			(int(enabled),),
		)
	log_admin_action(admin_user_id, "Changed external AI provider access", "AI Model Management", "service_key=ai_assistant")


def admin_analytics(admin_user_id: int) -> dict[str, Any]:
	require_admin(admin_user_id)
	with connection() as database:
		counts = database.execute(
			"""SELECT
				(SELECT COUNT(*) FROM users) AS users,
				(SELECT COUNT(*) FROM users WHERE is_active = 1) AS active_users,
				(SELECT COUNT(*) FROM users WHERE created_at >= datetime('now', '-30 days')) AS new_users,
				(SELECT COUNT(*) FROM resume_analyses) AS resume_analyses,
				(SELECT AVG(overall) FROM ats_scores) AS average_ats,
				(SELECT COUNT(*) FROM learning_courses WHERE active = 1) AS courses,
				(SELECT COUNT(*) FROM resources WHERE active = 1) AS resources,
				(SELECT COUNT(*) FROM learning_user_course_progress WHERE status = 'Completed') AS course_completions,
				(SELECT COUNT(*) FROM practice_attempts WHERE lower(kind) LIKE '%quiz%' OR lower(kind) LIKE '%mcq%') AS quiz_attempts,
				(SELECT COUNT(*) FROM practice_attempts WHERE lower(kind) LIKE '%coding%' OR lower(kind) LIKE '%challenge%') AS coding_attempts,
				(SELECT COUNT(*) FROM practice_attempts WHERE lower(kind) LIKE '%interview%') AS mock_interviews,
				(SELECT COUNT(*) FROM user_career_recommendations) AS career_recommendations,
				(SELECT COUNT(*) FROM generated_reports) AS reports_generated,
				(SELECT COUNT(*) FROM user_feedback) AS feedback_count,
				(SELECT COUNT(*) FROM admin_report_logs) AS admin_reports_generated"""
		).fetchone()
		user_growth = [dict(row) for row in database.execute(
			"SELECT date(created_at) AS day, COUNT(*) AS users FROM users GROUP BY date(created_at) ORDER BY day"
		).fetchall()]
		active_growth = [dict(row) for row in database.execute(
			"""SELECT date(created_at) AS day, COUNT(DISTINCT user_id) AS active_users
			   FROM login_audit WHERE was_successful = 1
			   GROUP BY date(created_at) ORDER BY day"""
		).fetchall()]
		resume_activity = [dict(row) for row in database.execute(
			"""SELECT date(created_at) AS day, COUNT(*) AS analyses, AVG(overall) AS average_ats
			   FROM ats_scores GROUP BY date(created_at) ORDER BY day"""
		).fetchall()]
		learning_activity = [dict(row) for row in database.execute(
			"""SELECT date(created_at) AS day, COUNT(*) AS updates, AVG(progress_percent) AS average_progress
			   FROM learning_activity GROUP BY date(created_at) ORDER BY day"""
		).fetchall()]
	readiness_values = [
		result["career_readiness"]
		for row in _admin_user_identity_rows(admin_user_id)
		if (result := get_jobs_report_progress(int(row["id"])))["career_readiness"] is not None
	]
	return {
		**dict(counts),
		"average_readiness": round(sum(readiness_values) / len(readiness_values), 1) if readiness_values else None,
		"user_growth": user_growth,
		"active_growth": active_growth,
		"resume_activity": resume_activity,
		"learning_activity": learning_activity,
	}


def _admin_user_identity_rows(admin_user_id: int) -> list[dict[str, Any]]:
	require_admin(admin_user_id)
	with connection() as database:
		return [dict(row) for row in database.execute("SELECT id FROM users").fetchall()]


def admin_report_data(
	admin_user_id: int,
	report_type: str,
	start_date: date,
	end_date: date,
	user_id: int | None,
) -> pd.DataFrame:
	require_admin(admin_user_id)
	start = start_date.isoformat()
	end = (end_date + timedelta(days=1)).isoformat()
	queries: dict[str, tuple[str, tuple[Any, ...]]] = {
		"User Report": (
			"""SELECT id AS user_id, full_name, email, role, is_active, created_at
			   FROM users WHERE date(created_at) BETWEEN ? AND ? ORDER BY created_at DESC""",
			(start, end),
		),
		"Learning Report": (
			"""SELECT u.id AS user_id, u.full_name, c.course_name, p.progress_percent, p.status, p.updated_at
			   FROM learning_user_course_progress p JOIN users u ON u.id = p.user_id
			   JOIN learning_courses c ON c.id = p.course_id
			   WHERE date(p.updated_at) BETWEEN ? AND ? ORDER BY p.updated_at DESC""",
			(start, end),
		),
		"Resume Analysis Report": (
			"""SELECT a.user_id, u.full_name, s.overall AS ats_score, a.created_at
			   FROM resume_analyses a JOIN users u ON u.id = a.user_id
			   JOIN ats_scores s ON s.analysis_id = a.id
			   WHERE date(a.created_at) BETWEEN ? AND ? ORDER BY a.created_at DESC""",
			(start, end),
		),
		"Career Recommendation Report": (
			"""SELECT r.user_id, u.full_name, r.career_name, r.recommendation_score, r.created_at
			   FROM user_career_recommendations r JOIN users u ON u.id = r.user_id
			   WHERE date(r.created_at) BETWEEN ? AND ? ORDER BY r.created_at DESC""",
			(start, end),
		),
		"Practice & Interview Report": (
			"""SELECT p.user_id, u.full_name, p.kind, p.subject, p.score, p.total, p.status, p.created_at
			   FROM practice_attempts p JOIN users u ON u.id = p.user_id
			   WHERE date(p.created_at) BETWEEN ? AND ? ORDER BY p.created_at DESC""",
			(start, end),
		),
		"Job/Internship Recommendation Report": (
			"""SELECT CAST(NULL AS INTEGER) AS user_id, CAST(NULL AS TEXT) AS recommended_role,
					  CAST(NULL AS INTEGER) AS recommendation_score, CAST(NULL AS TEXT) AS created_at
			   WHERE date('now') BETWEEN ? AND ? AND 0 ORDER BY created_at""",
			(start, end),
		),
		"Feedback Report": (
			"""SELECT f.id AS feedback_id, u.full_name, f.feedback_type, f.message, f.rating, f.status, f.created_at
			   FROM user_feedback f JOIN users u ON u.id = f.user_id
			   WHERE date(f.created_at) BETWEEN ? AND ? ORDER BY f.created_at DESC""",
			(start, end),
		),
		"Platform Activity Report": (
			"""SELECT a.user_id, u.full_name, a.module, a.activity, a.status, a.created_at
			   FROM activity_history a JOIN users u ON u.id = a.user_id
			   WHERE date(a.created_at) BETWEEN ? AND ? ORDER BY a.created_at DESC""",
			(start, end),
		),
	}
	if report_type not in queries:
		raise ValueError("Choose a supported report type.")
	query, params = queries[report_type]
	if user_id is not None and report_type != "Job/Internship Recommendation Report":
		user_columns = {
			"User Report": "id",
			"Learning Report": "p.user_id",
			"Resume Analysis Report": "a.user_id",
			"Career Recommendation Report": "r.user_id",
			"Practice & Interview Report": "p.user_id",
			"Job/Internship Recommendation Report": "user_id",
			"Feedback Report": "f.user_id",
			"Platform Activity Report": "a.user_id",
		}
		query = query.replace(" ORDER BY ", f" AND {user_columns[report_type]} = ? ORDER BY ", 1)
		params = (*params, user_id)
	with connection() as database:
		rows = database.execute(query + " LIMIT 500", params).fetchall()
	return pd.DataFrame([dict(row) for row in rows])


def admin_log_report(admin_user_id: int, report_type: str, start_date: date, end_date: date) -> None:
	require_admin(admin_user_id)
	with connection() as database:
		database.execute(
			"INSERT INTO admin_report_logs (admin_user_id, report_type, start_date, end_date) VALUES (?, ?, ?, ?)",
			(admin_user_id, report_type[:120], start_date.isoformat(), end_date.isoformat()),
		)
	log_admin_action(admin_user_id, "Generated platform report", "Reports", report_type)


def request_reset(email: str) -> tuple[str, str | None]:
	user = get_user(email)
	message = "If an account exists for that email, reset instructions are ready."
	if not user:
		return message, None
	reset_token = new_token()
	create_token("password_reset_tokens", user["id"], reset_token, expiry(minutes=RESET_TOKEN_MINUTES))
	return message, reset_token


def update_password(raw_token: str, password: str, confirmation: str) -> tuple[bool, str]:
	record = valid_token("password_reset_tokens", raw_token)
	if not record:
		return False, "This reset token is invalid or expired."
	errors = password_errors(password)
	if errors:
		return False, "Your password needs " + ", ".join(errors) + "."
	if password != confirmation:
		return False, "Passwords do not match."
	with connection() as database:
		database.execute(
			"UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
			(hash_password(password), record["user_id"]),
		)
	consume_token("password_reset_tokens", record["id"])
	return True, "Password updated. You can now log in."


def verify_email(raw_token: str) -> tuple[bool, str]:
	record = valid_token("email_verification_tokens", raw_token)
	if not record:
		return False, "This verification token is invalid or expired."
	with connection() as database:
		database.execute("UPDATE users SET email_verified = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (record["user_id"],))
	consume_token("email_verification_tokens", record["id"])
	return True, "Email verified successfully."


# -----------------------------------------------------------------------------
# Resume storage, analysis, scoring, and report services
# -----------------------------------------------------------------------------

SKILL_CATALOG = {
	"Programming Languages": ["Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Golang", "Rust", "PHP", "Ruby", "Kotlin", "Swift", "SQL", "R"],
	"Frameworks": ["Django", "Flask", "FastAPI", "React", "Angular", "Vue", "Next.js", "Node.js", "Express", "Spring", "ASP.NET", "Streamlit"],
	"Libraries": ["Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly", "Requests", "Scikit-learn", "OpenCV", "TensorFlow", "PyTorch", "Keras", "NLTK", "spaCy"],
	"Databases": ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Oracle", "Microsoft SQL Server", "Elasticsearch"],
	"AI/ML": ["Machine Learning", "Deep Learning", "Natural Language Processing", "NLP", "Computer Vision", "Generative AI", "Large Language Models", "LLM", "Transformers", "Prompt Engineering"],
	"Cloud": ["AWS", "Amazon Web Services", "Azure", "Google Cloud", "GCP", "Firebase", "Cloudflare"],
	"DevOps": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions", "Terraform", "Ansible", "CI/CD", "Linux"],
	"Tools": ["Git", "GitHub", "GitLab", "Jira", "Postman", "Figma", "Power BI", "Tableau", "Microsoft Excel", "Excel"],
	"Other IT Skills": ["HTML", "CSS", "REST APIs", "Data Structures", "Testing", "Statistics", "Accessibility", "API Design"],
}
SOFT_SKILL_CATALOG = ["Leadership", "Communication", "Problem Solving", "Problem-Solving", "Teamwork", "Collaboration", "Adaptability", "Critical Thinking", "Time Management", "Mentoring", "Stakeholder Management"]

CAREER_SKILLS = {
	"data scientist": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Statistics", "Machine Learning"],
	"machine learning engineer": ["Python", "SQL", "Scikit-learn", "TensorFlow", "PyTorch", "Docker", "Machine Learning"],
	"data analyst": ["SQL", "Python", "Microsoft Excel", "Power BI", "Tableau", "Statistics"],
	"software engineer": ["Git", "SQL", "Data Structures", "Testing", "API Design"],
	"web developer": ["HTML", "CSS", "JavaScript", "React", "Git", "API Design"],
	"backend developer": ["Python", "SQL", "REST APIs", "Git", "Docker", "PostgreSQL"],
	"frontend developer": ["HTML", "CSS", "JavaScript", "React", "TypeScript", "Accessibility"],
	"cloud engineer": ["Linux", "AWS", "Docker", "Kubernetes", "Terraform", "CI/CD"],
}

CAREER_DATASET = {
	"Python Developer": {
		"description": "Build backend services, APIs, automation scripts, and product features using Python and modern frameworks.",
		"required_skills": ["Python", "SQL", "Git", "REST APIs", "Django", "Flask", "FastAPI", "Docker", "PostgreSQL", "Testing"],
		"optional_skills": ["Linux", "Redis", "Celery", "AWS"],
		"related_technologies": ["Python", "Django", "FastAPI", "SQL", "Docker"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Python syntax", "OOP", "functions", "debugging"], "skills": ["Python", "Git"], "projects": ["CLI tools", "automation scripts"], "outcome": "Write and debug Python scripts confidently.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["Data structures", "APIs", "databases"], "skills": ["SQL", "REST APIs", "PostgreSQL"], "projects": ["CRUD API", "database-backed app"], "outcome": "Build real-world service logic.", "status": "Progressing"},
			{"stage": "Stage 3 → Frameworks", "topics": ["FastAPI", "Flask", "Django"], "skills": ["FastAPI", "Flask", "Django"], "projects": ["Task manager API", "blog service"], "outcome": "Develop production-ready backend services.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Deployment", "testing", "async work"], "skills": ["Testing", "Docker", "Linux"], "projects": ["Portfolio backend", "deployment pipeline"], "outcome": "Ship and maintain deployable products.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["System design", "problem solving", "OOP practice"], "skills": ["Problem Solving", "System Design"], "projects": ["Mock interviews", "API design practice"], "outcome": "Prepare for backend engineering interviews.", "status": "Planned"},
		],
		"learning_topics": ["Python Advanced", "REST APIs", "FastAPI", "SQL", "Docker", "Testing"],
		"projects": ["Task management API", "Portfolio data dashboard", "Automation script suite"],
		"future_roles": ["Backend Developer", "AI Engineer", "Data Engineer"],
		"learning_direction": "Focus on Python fundamentals, API design, database workflows, and deployable backend projects.",
		"overview": "A Python Developer creates software systems, automation, and APIs using Python, databases, and deployment tools."
	},
	"AI Engineer": {
		"description": "Design intelligent product features, integrate ML workflows, and manage AI-driven application services.",
		"required_skills": ["Python", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "FastAPI", "SQL", "Git", "Docker"],
		"optional_skills": ["Transformers", "MLOps", "AWS", "Prompt Engineering"],
		"related_technologies": ["Python", "Machine Learning", "Deep Learning", "NLP", "FastAPI"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Python", "Math", "statistics"], "skills": ["Python", "Statistics"], "projects": ["Data analysis notebook"], "outcome": "Build statistical and coding confidence.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["Machine Learning", "Numpy", "Pandas"], "skills": ["Machine Learning", "NumPy", "Pandas"], "projects": ["Prediction model"], "outcome": "Train and evaluate ML models.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Deep Learning", "NLP", "transformers"], "skills": ["Deep Learning", "NLP", "Transformers"], "projects": ["Sentiment classifier", "document summarizer"], "outcome": "Handle modern AI and language tasks.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Model deployment", "APIs"], "skills": ["FastAPI", "Docker"], "projects": ["AI application API"], "outcome": "Deploy AI solutions that users can access.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["Model evaluation", "system design", "MLOps"], "skills": ["Problem Solving", "Model Evaluation"], "projects": ["Case interviews", "AI portfolio project"], "outcome": "Prepare for technical AI interviews.", "status": "Planned"},
		],
		"learning_topics": ["Python Advanced", "Machine Learning", "Deep Learning", "NLP", "FastAPI", "MLOps"],
		"projects": ["Recommendation engine", "AI chatbot", "document intelligence app"],
		"future_roles": ["ML Engineer", "AI Research Engineer", "Applied Scientist"],
		"learning_direction": "Learn the ML pipeline, model evaluation, deep learning, and deployment through portfolio projects.",
		"overview": "An AI Engineer builds products that combine machine learning, NLP, deployable APIs, and product-focused experimentation."
	},
	"Machine Learning Engineer": {
		"description": "Build, evaluate, and deploy machine learning systems in production environments.",
		"required_skills": ["Python", "Machine Learning", "SQL", "Scikit-learn", "TensorFlow", "PyTorch", "Docker", "Git", "Statistics"],
		"optional_skills": ["MLOps", "Cloud", "FastAPI", "Kubernetes"],
		"related_technologies": ["Python", "Scikit-learn", "TensorFlow", "SQL", "Docker"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Python", "statistics", "linear algebra"], "skills": ["Python", "Statistics"], "projects": ["Regression analysis notebook"], "outcome": "Understand data science foundations.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["scikit-learn", "model selection", "feature engineering"], "skills": ["Scikit-learn", "SQL", "Machine Learning"], "projects": ["Classification model"], "outcome": "Create trainable models and evaluate them.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["deep learning", "distributed systems", "monitoring"], "skills": ["PyTorch", "TensorFlow", "Docker"], "projects": ["Model API service"], "outcome": "Move from notebooks to scalable services.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["MLOps", "CI/CD"], "skills": ["Git", "Docker", "Model Deployment"], "projects": ["Production ML system"], "outcome": "Operate reliable ML systems.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["ML case interviews", "evaluation metrics"], "skills": ["Problem Solving", "Statistics"], "projects": ["Mock case study"], "outcome": "Demonstrate strong ML decision-making.", "status": "Planned"},
		],
		"learning_topics": ["Machine Learning", "Scikit-learn", "Model Deployment", "Docker", "Statistics"],
		"projects": ["Price prediction system", "fraud detection model", "customer churn model"],
		"future_roles": ["AI Engineer", "Applied Scientist", "MLOps Engineer"],
		"learning_direction": "Strengthen ML fundamentals, model evaluation, and deployment quality for production-ready systems.",
		"overview": "A Machine Learning Engineer turns data into working prediction systems and brings them into production with engineering discipline."
	},
	"Data Scientist": {
		"description": "Analyze business and product data, build predictive models, and tell clear stories from complex information.",
		"required_skills": ["Python", "SQL", "Statistics", "Pandas", "NumPy", "Machine Learning", "Excel", "Tableau", "Data Visualization"],
		"optional_skills": ["Power BI", "Scikit-learn", "AWS"],
		"related_technologies": ["Python", "SQL", "Pandas", "Tableau", "Power BI"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Python", "statistics", "Excel"], "skills": ["Python", "Statistics", "Excel"], "projects": ["Sales data summary"], "outcome": "Work with data confidently.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["Pandas", "SQL", "visualization"], "skills": ["Pandas", "SQL", "Tableau"], "projects": ["Executive dashboard"], "outcome": "Clean data and build strong analysis narratives.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Machine Learning", "A/B testing"], "skills": ["Machine Learning", "Statistics"], "projects": ["Retention model"], "outcome": "Build predictive solutions with business context.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Business storytelling", "stakeholder communication"], "skills": ["Communication", "Visualization"], "projects": ["Product analytics case study"], "outcome": "Translate insights into recommendations.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["Metrics", "case studies", "SQL interviews"], "skills": ["Problem Solving", "SQL"], "projects": ["Case interview practice"], "outcome": "Perform well in business and technical interviews.", "status": "Planned"},
		],
		"learning_topics": ["Python", "SQL", "Statistics", "Pandas", "Data Visualization", "Machine Learning"],
		"projects": ["Customer segmentation", "marketing funnel analysis", "revenue forecasting model"],
		"future_roles": ["Senior Data Scientist", "Analytics Lead", "ML Product Analyst"],
		"learning_direction": "Balance data analysis, statistical reasoning, business communication, and model-building practice.",
		"overview": "A Data Scientist interprets datasets, tests hypotheses, builds models, and helps teams make evidence-based decisions."
	},
	"Data Analyst": {
		"description": "Turn raw data into clear trends, reports, and operational insights for decision-making.",
		"required_skills": ["SQL", "Python", "Excel", "Power BI", "Tableau", "Statistics", "Data Visualization"],
		"optional_skills": ["Pandas", "Machine Learning", "ETL"],
		"related_technologies": ["SQL", "Excel", "Power BI", "Tableau", "Python"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Excel", "data cleaning", "basic statistics"], "skills": ["Excel", "Statistics"], "projects": ["Sales report"], "outcome": "Clean and analyze data accurately.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["SQL", "Pandas", "charts"], "skills": ["SQL", "Python", "Data Visualization"], "projects": ["KPI dashboard"], "outcome": "Build repeatable analysis workflows.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Power BI", "Tableau", "storytelling"], "skills": ["Power BI", "Tableau"], "projects": ["Business dashboard"], "outcome": "Turn analysis into dashboard-led decisions.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Metrics", "A/B testing"], "skills": ["Communication", "Statistics"], "projects": ["Performance review dashboard"], "outcome": "Recommend actionable next steps.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["SQL queries", "metrics interpretation", "case interviews"], "skills": ["SQL", "Problem Solving"], "projects": ["Interview case set"], "outcome": "Confidently discuss analysis decisions.", "status": "Planned"},
		],
		"learning_topics": ["SQL", "Excel", "Power BI", "Data Visualization", "Statistics"],
		"projects": ["Dashboard design", "customer retention analysis", "sales analytics case"],
		"future_roles": ["Senior Analyst", "Business Intelligence Analyst", "Product Analyst"],
		"learning_direction": "Focus on SQL, analytical storytelling, dashboarding, and business problem solving.",
		"overview": "A Data Analyst translates operational data into insights that inform pricing, products, growth, and performance decisions."
	},
	"Backend Developer": {
		"description": "Design, build, and optimize server-side systems, APIs, and application logic.",
		"required_skills": ["Python", "SQL", "REST APIs", "Git", "Docker", "PostgreSQL", "JavaScript", "Node.js", "Testing"],
		"optional_skills": ["Redis", "Microservices", "AWS"],
		"related_technologies": ["Python", "SQL", "REST APIs", "Docker", "PostgreSQL"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Programming basics", "databases", "system design"], "skills": ["Python", "SQL"], "projects": ["API starter project"], "outcome": "Understand server-side architecture.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["REST APIs", "auth", "database design"], "skills": ["REST APIs", "PostgreSQL", "Git"], "projects": ["User API service"], "outcome": "Build robust backend services.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Scalability", "microservices", "caching"], "skills": ["Docker", "Redis"], "projects": ["Microservice project"], "outcome": "Prepare for production pipelines.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Testing", "security", "deployment"], "skills": ["Testing", "Docker", "Security"], "projects": ["Production-ready backend"], "outcome": "Ship maintainable systems.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["System design", "API interview questions"], "skills": ["System Design", "Problem Solving"], "projects": ["Mock interviews"], "outcome": "Perform well in technical interviews.", "status": "Planned"},
		],
		"learning_topics": ["REST APIs", "SQL", "Docker", "System Design", "Testing"],
		"projects": ["Inventory API", "user authentication service", "booking backend"],
		"future_roles": ["Senior Backend Engineer", "Platform Engineer", "Software Architect"],
		"learning_direction": "Build API architecture, backend reliability, and production deployment experience.",
		"overview": "A Backend Developer delivers the server-side logic that powers applications, APIs, and business workflows."
	},
	"Frontend Developer": {
		"description": "Create responsive user experiences and interactive web interfaces with modern UI tooling.",
		"required_skills": ["HTML", "CSS", "JavaScript", "React", "TypeScript", "Git", "Accessibility"],
		"optional_skills": ["Next.js", "UI/UX", "Design Systems"],
		"related_technologies": ["HTML", "CSS", "JavaScript", "React", "TypeScript"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["HTML", "CSS", "JavaScript basics"], "skills": ["HTML", "CSS", "JavaScript"], "projects": ["Landing page"], "outcome": "Build basic interfaces.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["React", "state management", "components"], "skills": ["React", "JavaScript"], "projects": ["Task app"], "outcome": "Create interactive UI flows.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["TypeScript", "responsive design", "performance"], "skills": ["TypeScript", "Accessibility"], "projects": ["Dashboard UI"], "outcome": "Ship polished, scalable interfaces.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Integration", "testing", "deployment"], "skills": ["Git", "Accessibility"], "projects": ["Portfolio website"], "outcome": "Deliver complete front-end experiences.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["UI architecture", "DOM", "problem solving"], "skills": ["Problem Solving", "JavaScript"], "projects": ["UI mock interviews"], "outcome": "Prepare for front-end technical interviews.", "status": "Planned"},
		],
		"learning_topics": ["HTML", "CSS", "JavaScript", "React", "TypeScript", "Accessibility"],
		"projects": ["Portfolio website", "dashboard UI", "e-commerce landing page"],
		"future_roles": ["Senior Frontend Engineer", "UI Engineer", "Product Designer (with engineering)"],
		"learning_direction": "Improve UI thinking, interaction design, accessibility, and front-end architecture decisions.",
		"overview": "A Frontend Developer builds engaging interfaces and experiences that connect design intent to product functionality."
	},
	"Full Stack Developer": {
		"description": "Work across frontend, backend, data, and deployment layers to deliver end-to-end products.",
		"required_skills": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL", "REST APIs", "Git", "Docker"],
		"optional_skills": ["TypeScript", "Flask", "FastAPI", "AWS"],
		"related_technologies": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["HTML", "CSS", "Python basics"], "skills": ["HTML", "CSS", "Python"], "projects": ["Static webpage"], "outcome": "Be comfortable across client and server basics.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["JavaScript", "React", "REST APIs"], "skills": ["JavaScript", "React", "REST APIs"], "projects": ["To-do app"], "outcome": "Link UI logic to backend services.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["database design", "deployment", "testing"], "skills": ["SQL", "Docker", "Testing"], "projects": ["Full app with API"], "outcome": "Produce complete product workflows.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Authentication", "security", "CI/CD"], "skills": ["Git", "Security"], "projects": ["Full-stack SaaS MVP"], "outcome": "Build and ship polished products.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["System design", "debugging", "full stack tooling"], "skills": ["Problem Solving", "System Design"], "projects": ["Full stack mock interview"], "outcome": "Handle product and engineering conversations confidently.", "status": "Planned"},
		],
		"learning_topics": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL", "Docker"],
		"projects": ["Inventory dashboard", "blog platform", "e-commerce MVP"],
		"future_roles": ["Senior Full Stack Engineer", "Tech Lead", "Product Engineer"],
		"learning_direction": "Build strong breadth across frontend, backend, database, and deployment skills.",
		"overview": "A Full Stack Developer handles both the user-facing interface and the underlying application and data systems."
	},
	"DevOps Engineer": {
		"description": "Enable scalable, reliable software delivery through automation, infrastructure, and operational excellence.",
		"required_skills": ["Linux", "Docker", "Kubernetes", "AWS", "CI/CD", "Git", "Terraform", "Shell Scripting", "Monitoring"],
		"optional_skills": ["Ansible", "Python", "Cloud"],
		"related_technologies": ["Linux", "Docker", "Kubernetes", "AWS", "Terraform"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Linux", "shell", "networking basics"], "skills": ["Linux", "Git"], "projects": ["Server setup lab"], "outcome": "Understand infrastructure basics.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["Docker", "CI/CD", "automation"], "skills": ["Docker", "CI/CD"], "projects": ["Pipeline automation"], "outcome": "Automate standard delivery workflows.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Kubernetes", "Terraform", "cloud"], "skills": ["Kubernetes", "Terraform", "AWS"], "projects": ["Deployment automation stack"], "outcome": "Handle cloud-native infrastructure.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Monitoring", "incident response"], "skills": ["Monitoring", "Linux"], "projects": ["Monitoring setup"], "outcome": "Operate and troubleshoot systems reliably.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["Linux systems", "deployment troubleshooting", "pipelines"], "skills": ["Problem Solving", "Linux"], "projects": ["Interview labs"], "outcome": "Demonstrate strong ops thinking.", "status": "Planned"},
		],
		"learning_topics": ["Linux", "Docker", "Kubernetes", "Terraform", "CI/CD", "AWS"],
		"projects": ["CI pipeline for app", "containerized deployment", "infrastructure automation"],
		"future_roles": ["Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
		"learning_direction": "Learn Linux, infrastructure automation, cloud systems, and end-to-end deployment reliability.",
		"overview": "A DevOps Engineer improves delivery speed, reliability, and operational consistency through automation and infrastructure discipline."
	},
	"Cloud Engineer": {
		"description": "Architect and maintain cloud-native applications, automation, and platform infrastructure for modern software delivery.",
		"required_skills": ["AWS", "Azure", "Linux", "Docker", "Kubernetes", "Terraform", "CI/CD", "Networking"],
		"optional_skills": ["Python", "Monitoring", "Security"],
		"related_technologies": ["AWS", "Azure", "Linux", "Kubernetes", "Terraform"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["Cloud basics", "Linux", "networking"], "skills": ["Linux", "Networking"], "projects": ["Cloud VM setup"], "outcome": "Understand cloud fundamentals.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["AWS", "Azure", "containers"], "skills": ["AWS", "Docker", "Kubernetes"], "projects": ["Container deployment"], "outcome": "Prepare for cloud deployment tasks.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["Terraform", "CI/CD", "monitoring"], "skills": ["Terraform", "CI/CD", "Monitoring"], "projects": ["Infrastructure platform"], "outcome": "Build scalable infrastructure with automation.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["Architecture", "resilience", "security"], "skills": ["Security", "Cloud"], "projects": ["Multi-service cloud deployment"], "outcome": "Design secure production systems.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["cloud architecture", "costing", "system design"], "skills": ["System Design", "Problem Solving"], "projects": ["Cloud interview prep"], "outcome": "Confidently discuss cloud architecture decisions.", "status": "Planned"},
		],
		"learning_topics": ["AWS", "Docker", "Kubernetes", "Terraform", "Linux", "CI/CD"],
		"projects": ["Cloud deployment workflow", "serverless app setup", "infrastructure automation"],
		"future_roles": ["Solutions Architect", "Platform Engineer", "Cloud Security Engineer"],
		"learning_direction": "Learn cloud architecture, automation, and secure infrastructure operations with measurable deployment practice.",
		"overview": "A Cloud Engineer builds and manages cloud systems, infrastructure pipelines, and resilient application environments."
	},
	"Cyber Security Analyst": {
		"description": "Protect systems, detect threats, and improve security posture through monitoring and analysis.",
		"required_skills": ["Linux", "Networking", "Python", "Security", "Threat Analysis", "Monitoring", "SQL", "Cloud"],
		"optional_skills": ["SIEM", "AWS", "Pentesting", "Security Tools"],
		"related_technologies": ["Linux", "Networking", "Security", "Cloud", "Python"],
		"roadmap_stages": [
			{"stage": "Stage 1 → Foundation", "topics": ["networking", "security basics", "Linux"], "skills": ["Linux", "Security"], "projects": ["Security lab setup"], "outcome": "Understand the security landscape.", "status": "Beginner"},
			{"stage": "Stage 2 → Core Technical Skills", "topics": ["threats", "log analysis", "endpoint basics"], "skills": ["Threat Analysis", "Monitoring"], "projects": ["Log analysis exercise"], "outcome": "Identify risks from system signals.", "status": "Progressing"},
			{"stage": "Stage 3 → Advanced Skills", "topics": ["cloud security", "incident response", "automation"], "skills": ["Cloud", "Python"], "projects": ["SOC workflow automation"], "outcome": "Improve security operations with tools and automation.", "status": "Planned"},
			{"stage": "Stage 4 → Projects", "topics": ["vulnerability analysis", "forensics"], "skills": ["Security", "Monitoring"], "projects": ["Security review project"], "outcome": "Demonstrate applied security skills.", "status": "Planned"},
			{"stage": "Stage 5 → Interview Preparation", "topics": ["security scenarios", "incident response"], "skills": ["Problem Solving", "Security"], "projects": ["Mock incident drills"], "outcome": "Prepare for telecom and security assessments.", "status": "Planned"},
		],
		"learning_topics": ["Linux", "Networking", "Security", "Threat Analysis", "Monitoring", "Cloud"],
		"projects": ["Threat log review", "security checklist project", "monitoring lab setup"],
		"future_roles": ["Security Engineer", "Threat Analyst", "SOC Lead"],
		"learning_direction": "Build security fundamentals, monitoring maturity, and operational response capability.",
		"overview": "A Cyber Security Analyst protects digital systems by reviewing signals, investigating threats, and supporting resilient security operations."
	},
}

SECTION_TERMS = {
	"Education": ("education", "academic", "university", "college"),
	"Technical skills": ("skills", "technical skills", "technologies", "tools"),
	"Projects": ("projects", "project experience", "selected work"),
	"Experience": ("experience", "employment", "work history", "professional experience"),
	"Certifications": ("certifications", "certificates", "licenses"),
	"Achievements": ("achievements", "awards", "accomplishments", "honors"),
	"Professional summary": ("summary", "profile", "objective", "about me"),
}


def resume_db() -> Iterator[sqlite3.Connection]:
	return connection()


def validate_resume_pdf(filename: str, content: bytes) -> None:
	if Path(filename).suffix.lower() != ".pdf":
		raise ValueError("Please upload a PDF resume.")
	if not content or len(content) > MAX_RESUME_BYTES:
		raise ValueError("Resume size must be between 1 byte and 10 MB.")
	if not content.startswith(b"%PDF-"):
		raise ValueError("The selected file is not a valid PDF document.")
	try:
		with fitz.open(stream=content, filetype="pdf") as document:
			if document.page_count == 0 or document.page_count > 80:
				raise ValueError("Resume PDF must contain between 1 and 80 pages.")
	except (fitz.FileDataError, fitz.EmptyFileError) as error:
		raise ValueError("The selected PDF could not be opened.") from error


def private_file_path(root: Path, storage_key: str, user_id: int) -> Path:
	parts = Path(storage_key).parts
	path = (root / storage_key).resolve()
	if not parts or parts[0] != str(user_id) or not path.is_relative_to(root.resolve()):
		raise PermissionError("This file is not available to the current account.")
	return path


def write_private_file(root: Path, user_id: int, content: bytes, suffix: str = "pdf") -> str:
	folder = root / str(user_id)
	folder.mkdir(parents=True, exist_ok=True)
	try:
		os.chmod(folder, 0o700)
	except OSError:
		pass
	storage_key = f"{user_id}/{secrets.token_hex(24)}.{suffix}"
	path = root / storage_key
	with path.open("xb") as output:
		output.write(content)
	try:
		os.chmod(path, 0o600)
	except OSError:
		pass
	return storage_key


def upload_resume(user_id: int, filename: str, content: bytes) -> int:
	validate_resume_pdf(filename, content)
	RESUME_STORAGE.mkdir(parents=True, exist_ok=True)
	storage_key = write_private_file(RESUME_STORAGE, user_id, content)
	try:
		with connection() as database:
			database.execute("UPDATE resumes SET is_current = 0 WHERE user_id = ?", (user_id,))
			cursor = database.execute(
				"INSERT INTO resumes (user_id, original_filename, storage_key, sha256, file_size) VALUES (?, ?, ?, ?, ?)",
				(user_id, Path(filename).name[:255], storage_key, hashlib.sha256(content).hexdigest(), len(content)),
			)
			return int(cursor.lastrowid)
	except Exception:
		private_file_path(RESUME_STORAGE, storage_key, user_id).unlink(missing_ok=True)
		raise


def get_current_resume(user_id: int) -> dict[str, Any] | None:
	with connection() as database:
		row = database.execute(
			"SELECT id, original_filename, file_size, sha256, uploaded_at FROM resumes WHERE user_id = ? AND is_current = 1 ORDER BY id DESC LIMIT 1",
			(user_id,),
		).fetchone()
	return dict(row) if row else None


def get_resume_bytes(user_id: int, resume_id: int) -> bytes:
	with connection() as database:
		row = database.execute("SELECT storage_key FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user_id)).fetchone()
	if not row:
		raise PermissionError("Resume not found for this account.")
	return private_file_path(RESUME_STORAGE, row["storage_key"], user_id).read_bytes()


def get_profile_context(user_id: int) -> dict[str, Any]:
	"""Read optional profile data without assuming a not-yet-present profile schema."""
	context: dict[str, Any] = {"career_goal": "", "skills": []}
	with connection() as database:
		tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
		for table in ("profiles", "user_profiles", "my_profiles"):
			if table not in tables:
				continue
			columns = {row[1] for row in database.execute(f"PRAGMA table_info({table})")}
			if "user_id" not in columns:
				continue
			row = database.execute(f"SELECT * FROM {table} WHERE user_id = ? LIMIT 1", (user_id,)).fetchone()
			if not row:
				continue
			for field in ("career_goal", "career_objective", "target_role"):
				if field in columns and row[field]:
					context["career_goal"] = str(row[field]).strip()
					break
			for field in ("skills", "technical_skills", "skills_json"):
				if field in columns and row[field]:
					value = row[field]
					try:
						context["skills"] = json.loads(value) if isinstance(value, str) else value
					except json.JSONDecodeError:
						context["skills"] = [item.strip() for item in str(value).split(",") if item.strip()]
					break
			break
	return context


def get_user_profile(user_id: int) -> dict[str, Any]:
	profile = {"career_goal": "", "education": "", "experience_summary": "", "skills": [], "interests": []}
	with connection() as database:
		row = database.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
	if row:
		profile["career_goal"] = row["career_goal"]
		profile["education"] = row["education"]
		profile["experience_summary"] = row["experience_summary"]
		profile["skills"] = json.loads(row["skills_json"])
		profile["interests"] = json.loads(row["interests_json"])
	return profile


def save_user_profile(user_id: int, profile: dict[str, Any]) -> None:
	with connection() as database:
		database.execute(
			"""INSERT INTO user_profiles (user_id, career_goal, education, experience_summary, skills_json, interests_json)
			   VALUES (?, ?, ?, ?, ?, ?)
			   ON CONFLICT(user_id) DO UPDATE SET career_goal = excluded.career_goal,
			       education = excluded.education, experience_summary = excluded.experience_summary,
			       skills_json = excluded.skills_json, interests_json = excluded.interests_json,
			       updated_at = CURRENT_TIMESTAMP""",
			(user_id, profile["career_goal"], profile["education"], profile["experience_summary"],
			 json.dumps(profile["skills"]), json.dumps(profile["interests"])),
		)
	record_activity(user_id, "Profile updated", "My Profile", "Career goal, education, experience, or skills were saved.")


def profile_completion(user: dict[str, Any], profile: dict[str, Any]) -> int:
	completed = sum(bool(value) for value in (
		user.get("full_name"), user.get("email"), profile["career_goal"], profile["education"],
		profile["experience_summary"], profile["skills"], profile["interests"],
	))
	return round(completed / 7 * 100)


def extract_pdf_text(content: bytes) -> str:
	with fitz.open(stream=content, filetype="pdf") as document:
		text = "\n".join(page.get_text("text") for page in document)
	return re.sub(r"[ \t]+", " ", text).strip()


def categorize_skills(text: str) -> list[dict[str, str]]:
	detected: list[dict[str, str]] = []
	for category, skills in SKILL_CATALOG.items():
		for skill in skills:
			pattern = rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])"
			if re.search(pattern, text, flags=re.IGNORECASE):
				detected.append({"skill": skill, "category": category, "source": "resume"})
	return detected


def extract_resume_data(text: str, full_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
	lines = [line.strip(" \t-•") for line in text.splitlines() if line.strip()]
	emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE)
	phones = re.findall(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)", text)
	sections: dict[str, bool] = {}
	lower = text.lower()
	for name, terms in SECTION_TERMS.items():
		sections[name] = any(term in lower for term in terms)

	def matching_lines(*terms: str) -> list[str]:
		return [line for line in lines if any(term in line.lower() for term in terms)][:8]

	section_counts = sum(sections.values())
	missing_sections = [name for name, present in sections.items() if not present]
	contact_name = next((line for line in lines[:5] if re.search(r"[A-Za-z]", line) and "@" not in line and not re.search(r"\d{3}", line)), full_name)
	word_count = len(re.findall(r"\b\w+\b", text))
	achievement_lines = [line for line in lines if re.search(r"\b(\d+%|\$\d+|increased|reduced|improved|achieved|awarded|ranked)\b", line, re.IGNORECASE)][:8]
	keywords = sorted(set(re.findall(r"\b(?:Python|JavaScript|Java|SQL|AWS|Azure|Docker|Kubernetes|machine learning|NLP|React|leadership|communication|analytics|automation|API|agile|cloud)\b", text, re.IGNORECASE)), key=str.lower)
	data = {
		"name": contact_name,
		"email": emails[0] if emails else "Not detected",
		"phone": phones[0].strip() if phones else "Not detected",
		"education": matching_lines("university", "college", "bachelor", "master", "b.sc", "m.sc", "degree", "education"),
		"projects": matching_lines("project", "built ", "developed ", "implemented "),
		"experience": matching_lines("experience", "engineer", "developer", "intern", "manager", "analyst", "employment"),
		"certifications": matching_lines("certification", "certified", "certificate", "license"),
		"achievements": achievement_lines,
		"keywords": keywords,
		"soft_skills": [skill for skill in SOFT_SKILL_CATALOG if re.search(rf"(?<![A-Za-z]){re.escape(skill)}(?![A-Za-z])", text, re.IGNORECASE)],
		"word_count": word_count,
		"strengths": [],
		"weak_areas": [],
		"missing_sections": missing_sections,
		"important_information": [item for item in ["Email detected" if emails else "Email not detected", "Phone detected" if phones else "Phone not detected", f"{word_count} words"]],
	}
	return data, {"available": [name for name, present in sections.items() if present], "missing": missing_sections, "detected": sections}


def compare_resume_to_job_description(resume_text: str, job_description: str, full_name: str) -> dict[str, Any]:
	def canonical_skill(skill: str) -> str:
		normalized = re.sub(r"[\s_]+", " ", skill.strip().lower()).replace("-", " ")
		aliases = {
			"golang": "go",
			"amazon web services": "aws",
			"google cloud": "gcp",
			"natural language processing": "nlp",
			"large language models": "llm",
		}
		return aliases.get(normalized, normalized)

	def unique_skills(text: str) -> dict[str, str]:
		skills: dict[str, str] = {}
		for item in categorize_skills(text):
			skills.setdefault(canonical_skill(item["skill"]), item["skill"])
		for skill in SOFT_SKILL_CATALOG:
			pattern = rf"(?<![A-Za-z]){re.escape(skill)}(?![A-Za-z])"
			if re.search(pattern, text, flags=re.IGNORECASE):
				skills.setdefault(canonical_skill(skill), skill)
		return skills

	resume_skills = unique_skills(resume_text)
	required_skills = unique_skills(job_description)
	matched = [skill for key, skill in required_skills.items() if key in resume_skills]
	missing = [skill for key, skill in required_skills.items() if key not in resume_skills]
	match_score = round(len(matched) / len(required_skills) * 100) if required_skills else None

	data, _ = extract_resume_data(resume_text, full_name)
	section_headings = {
		" ".join(name.casefold().split())
		for name in SECTION_TERMS
	}
	section_headings.update(" ".join(term.casefold().split()) for terms in SECTION_TERMS.values() for term in terms)
	experience = [
		line for line in data["experience"]
		if " ".join(line.strip(" :|-").casefold().split()) not in section_headings
	]
	projects = [
		line for line in data["projects"]
		if " ".join(line.strip(" :|-").casefold().split()) not in section_headings
	]
	stop_words = {
		"and", "are", "for", "from", "have", "has", "into", "our", "the", "their",
		"this", "that", "they", "using", "with", "will", "you", "your", "work",
		"working", "years", "year", "role", "team", "strong", "ability", "including",
	}
	job_terms = {
		word for word in re.findall(r"[a-z][a-z+#.]{2,}", job_description.lower())
		if word not in stop_words
	}
	evidence: list[dict[str, Any]] = []
	for source, lines in (("Experience", experience), ("Projects", projects)):
		for line in lines:
			line_terms = set(re.findall(r"[a-z][a-z+#.]{2,}", line.lower()))
			overlap = sorted(line_terms & job_terms)
			if overlap:
				evidence.append({"source": source, "line": line, "matched_terms": overlap})
	evidence.sort(key=lambda item: len(item["matched_terms"]), reverse=True)

	suggestions: list[str] = []
	for skill in missing[:8]:
		suggestions.append(
			f"If you have used {skill}, add a specific project or work example showing how; "
			"otherwise, treat it as a learning gap rather than claiming it."
		)
	if not matched and required_skills:
		suggestions.append(
			"Add a clearly labeled Skills section containing only technologies you can demonstrate; "
			"the current resume text has no catalogue-recognized skill overlap with this description."
		)
	if not evidence and (experience or projects):
		suggestions.append(
			"Rewrite the most relevant experience or project bullets with the same accurate role terminology "
			"used in the job description, and include verifiable outcomes where possible."
		)
	if not experience and not projects:
		suggestions.append(
			"Add relevant project or work examples that show how you applied your skills; include your role "
			"and a measurable result when you can support it."
		)
	if not suggestions:
		suggestions.append(
			"Prioritize the experience and project evidence that best supports this role, and keep every claim verifiable."
		)
	return {
		"required_skills": list(required_skills.values()),
		"resume_skills": list(resume_skills.values()),
		"matched_skills": matched,
		"missing_skills": missing,
		"match_score": match_score,
		"experience": experience,
		"projects": projects,
		"evidence": evidence[:6],
		"suggestions": suggestions,
	}


def score_resume(text: str, data: dict[str, Any], detected_skills: list[dict[str, str]]) -> dict[str, int]:
	word_count = data["word_count"]
	sections = data["missing_sections"]
	skill_score = min(100, round(len(detected_skills) / 12 * 100))
	keyword_score = min(100, len(data["keywords"]) * 14)
	structure_score = min(100, max(0, (7 - len(sections)) * 14))
	education_score = 100 if data["education"] else (30 if re.search(r"\b(university|degree|college|bachelor|master)\b", text, re.I) else 0)
	experience_score = min(100, 35 + 20 * bool(data["experience"]) + 20 * bool(data["achievements"]) + 25 * bool(re.search(r"\b(19|20)\d{2}\b", text)))
	project_score = min(100, 30 + 35 * bool(data["projects"]) + 35 * bool(data["achievements"]))
	completeness_score = min(100, round((7 - len(sections)) / 7 * 100))
	if word_count < 150:
		completeness_score = min(completeness_score, 35)
	elif word_count > 1000:
		completeness_score = min(completeness_score, 75)
	structure_score = min(100, structure_score + (10 if data["email"] != "Not detected" else 0) + (5 if data["phone"] != "Not detected" else 0))
	components = {
		"skills": skill_score,
		"keywords": keyword_score,
		"structure": structure_score,
		"education": education_score,
		"experience": experience_score,
		"projects": project_score,
		"completeness": completeness_score,
	}
	components["overall"] = round(sum(components.values()) / len(components))
	return components


def analyze_resume_text(text: str, full_name: str, profile: dict[str, Any]) -> dict[str, Any]:
	data, sections = extract_resume_data(text, full_name)
	detected = categorize_skills(text)
	profile_skills = profile.get("skills") or []
	if isinstance(profile_skills, str):
		profile_skills = [item.strip() for item in profile_skills.split(",") if item.strip()]
	goal = str(profile.get("career_goal") or "").strip()
	goal_key = next((key for key in CAREER_SKILLS if key in goal.lower()), None)
	recommended = CAREER_SKILLS.get(goal_key, []) if goal_key else []
	data["career_goal"] = goal or "Not provided"
	detected_names = {item["skill"].lower() for item in detected}
	missing = [skill for skill in recommended if skill.lower() not in detected_names]
	profile_items = [{"skill": str(skill), "category": "Profile skill", "source": "profile"} for skill in profile_skills if str(skill).strip()]
	gap_items = [{"skill": skill, "category": "Career goal", "source": "missing"} for skill in missing]
	suggestions: list[dict[str, str]] = []
	if not sections["available"] or not data["word_count"]:
		suggestions.append({"priority": "High", "text": "This PDF yielded little readable text. Export a text-based PDF rather than a scanned image so ATS tools can read it."})
	for section in data["missing_sections"]:
		suggestions.append({"priority": "High" if section in {"Experience", "Technical skills", "Education"} else "Medium", "text": f"Add a clearly labeled {section} section; it was not detected in this resume."})
	if len(detected) < 5:
		suggestions.append({"priority": "Medium", "text": "Your resume shows few recognizable technical skills. Add a concise skills section using the exact tools and technologies demonstrated in your work."})
	if not data["achievements"] and data["experience"]:
		suggestions.append({"priority": "High", "text": "Experience is present, but measurable outcomes were not detected. Add verified quantities such as time saved, users supported, or delivery impact."})
	if data["word_count"] < 150:
		suggestions.append({"priority": "High", "text": "The extracted resume is unusually brief. Expand relevant project and experience details, and check whether the PDF is image-only."})
	if data["word_count"] > 1000:
		suggestions.append({"priority": "Medium", "text": "The resume is lengthy. Prioritize the most role-relevant experience and remove repeated or dated detail."})
	if not data["keywords"]:
		suggestions.append({"priority": "Medium", "text": "Few role-related keywords were detected. Add accurate terms from the target job description where they match your actual experience."})
	if goal and not goal_key:
		suggestions.append({"priority": "Info", "text": f"Career goal '{goal}' does not yet match a configured role map. No career-specific missing skills were inferred."})
	if missing:
		suggestions.append({"priority": "Medium", "text": "For your stated career goal, consider demonstrating these skills through coursework or projects: " + ", ".join(missing[:6]) + "."})
	if not suggestions:
		suggestions.append({"priority": "Info", "text": "The detected sections are broadly complete. Tailor the summary and evidence to each target role, using only claims you can support."})
	data["strengths"] = [
		f"{len(detected)} recognizable technical skills detected" if detected else "Resume content is available for analysis",
		f"{len(sections['available'])} of {len(SECTION_TERMS)} common sections detected",
	]
	data["weak_areas"] = [item["text"] for item in suggestions if item["priority"] == "High"]
	return {"data": data, "sections": sections, "detected_skills": detected, "profile_skills": profile_items, "missing_skills": gap_items, "recommended_skills": [{"skill": skill, "category": "Career goal", "source": "recommended"} for skill in recommended], "scores": score_resume(text, data, detected), "suggestions": suggestions, "career_goal": goal}


def persist_resume_analysis(user_id: int, resume_id: int, result: dict[str, Any]) -> int:
	skill_rows = result["detected_skills"] + result["profile_skills"] + result["missing_skills"] + result["recommended_skills"]
	with connection() as database:
		owned = database.execute("SELECT id FROM resumes WHERE id = ? AND user_id = ?", (resume_id, user_id)).fetchone()
		if not owned:
			raise PermissionError("Resume not found for this account.")
		cursor = database.execute(
			"INSERT INTO resume_analyses (user_id, resume_id, extracted_data_json, section_data_json) VALUES (?, ?, ?, ?)",
			(user_id, resume_id, json.dumps(result["data"]), json.dumps(result["sections"])),
		)
		analysis_id = int(cursor.lastrowid)
		score_keys = ("overall", "skills", "keywords", "structure", "education", "experience", "projects", "completeness")
		database.execute(
			"INSERT INTO ats_scores (user_id, analysis_id, " + ", ".join(score_keys) + ") VALUES (?, ?, " + ", ".join("?" for _ in score_keys) + ")",
			(user_id, analysis_id, *(result["scores"][key] for key in score_keys)),
		)
		database.executemany(
			"INSERT OR IGNORE INTO extracted_skills (user_id, analysis_id, skill, category, source) VALUES (?, ?, ?, ?, ?)",
			[(user_id, analysis_id, row["skill"], row["category"], row["source"]) for row in skill_rows],
		)
		database.executemany(
			"INSERT INTO resume_suggestions (user_id, analysis_id, priority, suggestion) VALUES (?, ?, ?, ?)",
			[(user_id, analysis_id, item["priority"], item["text"]) for item in result["suggestions"]],
		)
	return analysis_id


def decode_analysis(row: sqlite3.Row) -> dict[str, Any]:
	result = dict(row)
	result["data"] = json.loads(result.pop("extracted_data_json"))
	result["sections"] = json.loads(result.pop("section_data_json"))
	return result


def get_analysis(user_id: int, analysis_id: int | None = None) -> dict[str, Any] | None:
	with connection() as database:
		if analysis_id is None:
			row = database.execute(
				"""SELECT a.id, a.resume_id, a.extracted_data_json, a.section_data_json, a.created_at,
				          r.original_filename, r.file_size, r.uploaded_at,
				          s.overall, s.skills, s.keywords, s.structure, s.education, s.experience, s.projects, s.completeness
				   FROM resume_analyses a JOIN resumes r ON r.id = a.resume_id
				   JOIN ats_scores s ON s.analysis_id = a.id
				   WHERE a.user_id = ? ORDER BY a.id DESC LIMIT 1""", (user_id,)
			).fetchone()
		else:
			row = database.execute(
				"""SELECT a.id, a.resume_id, a.extracted_data_json, a.section_data_json, a.created_at,
				          r.original_filename, r.file_size, r.uploaded_at,
				          s.overall, s.skills, s.keywords, s.structure, s.education, s.experience, s.projects, s.completeness
			   FROM resume_analyses a JOIN resumes r ON r.id = a.resume_id
			   JOIN ats_scores s ON s.analysis_id = a.id
			   WHERE a.user_id = ? AND a.id = ?""", (user_id, analysis_id)
			).fetchone()
		if not row:
			return None
		result = decode_analysis(row)
		result["skills_list"] = [dict(item) for item in database.execute(
			"SELECT skill, category, source FROM extracted_skills WHERE user_id = ? AND analysis_id = ? ORDER BY category, skill", (user_id, row["id"])
		).fetchall()]
		result["suggestions"] = [dict(item) for item in database.execute(
			"SELECT priority, suggestion AS text FROM resume_suggestions WHERE user_id = ? AND analysis_id = ? ORDER BY id", (user_id, row["id"])
		).fetchall()]
	return result


def list_resume_history(user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute(
			"""SELECT a.id, a.created_at, r.original_filename, s.overall
			   FROM resume_analyses a JOIN resumes r ON r.id = a.resume_id
			   JOIN ats_scores s ON s.analysis_id = a.id
			   WHERE a.user_id = ? ORDER BY a.id DESC LIMIT 30""", (user_id,)
		).fetchall()
	return [dict(row) for row in rows]


def get_dashboard_resume_summary(user_id: int) -> dict[str, Any]:
	"""Stable integration contract for a future Dashboard's score and activity cards."""
	with connection() as database:
		score = database.execute("SELECT overall, created_at FROM ats_scores WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
		activity = database.execute(
			"""SELECT 'Resume analyzed' AS label, r.original_filename AS detail, a.created_at AS occurred_at
			   FROM resume_analyses a JOIN resumes r ON r.id = a.resume_id
			   WHERE a.user_id = ? ORDER BY a.id DESC LIMIT 5""", (user_id,)
		).fetchall()
	return {"latest_ats_score": dict(score) if score else None, "recent_activity": [dict(row) for row in activity]}


def seed_career_dataset() -> None:
	with connection() as database:
		for career_name, payload in CAREER_DATASET.items():
			database.execute(
				"INSERT OR IGNORE INTO career_roles (career_name, description, overview, learning_direction, future_roles_json) VALUES (?, ?, ?, ?, ?)",
				(career_name, payload["description"], payload["overview"], payload["learning_direction"], json.dumps(payload["future_roles"])),
			)
			role_row = database.execute("SELECT id FROM career_roles WHERE career_name = ?", (career_name,)).fetchone()
			if not role_row:
				continue
			career_id = role_row["id"]
			for skill in payload["required_skills"]:
				database.execute(
					"INSERT OR IGNORE INTO career_required_skills (career_id, skill, category, is_optional) VALUES (?, ?, ?, 0)",
					(career_id, skill, "Required"),
				)
			for skill in payload["optional_skills"]:
				database.execute(
					"INSERT OR IGNORE INTO career_required_skills (career_id, skill, category, is_optional) VALUES (?, ?, ?, 1)",
					(career_id, skill, "Optional"),
				)
			for stage in payload["roadmap_stages"]:
				database.execute(
					"INSERT OR IGNORE INTO career_roadmaps (career_id, stage_name, topics_json, skills_json, projects_json, outcome, progress_label) VALUES (?, ?, ?, ?, ?, ?, ?)",
					(career_id, stage["stage"], json.dumps(stage["topics"]), json.dumps(stage["skills"]), json.dumps(stage["projects"]), stage["outcome"], stage["status"]),
				)
			for topic in payload["learning_topics"]:
				database.execute(
					"INSERT OR IGNORE INTO career_learning_paths (career_id, recommended_topic, reason, priority, estimated_stage, related_skills_json) VALUES (?, ?, ?, ?, ?, ?)",
					(career_id, topic, f"Strengthens the core path for {career_name} and aligns with the roadmap.", "High" if topic in ["Python Advanced", "Machine Learning", "Deep Learning"] else "Medium", "Foundation" if topic in ["Python", "SQL", "HTML", "CSS"] else "Core", json.dumps([skill for skill in payload["required_skills"][:4]])),
				)


def learning_topic(title: str, explanation: str, notes: list[str], examples: list[str], questions: list[str], status: str = "Not started") -> dict[str, Any]:
	return {
		"title": title,
		"explanation": explanation,
		"notes": notes,
		"examples": examples,
		"questions": questions,
		"status": status,
	}


def learning_project(title: str, difficulty: str, technologies: list[str], objective: str, required_skills: list[str], steps: list[str], outcome: str) -> dict[str, Any]:
	return {
		"title": title,
		"difficulty": difficulty,
		"technologies": technologies,
		"objective": objective,
		"required_skills": required_skills,
		"steps": steps,
		"expected_outcome": outcome,
	}


LEARNING_HUB_COURSE_DATA: dict[str, dict[str, Any]] = {
	"Python": {
		"description": "Build a strong foundation in Python for automation, backend work, and AI-driven products.",
		"difficulty": "Beginner",
		"duration": "4 weeks",
		"icon": ":material/code:",
		"topics": [
			learning_topic("Python Basics", "Understand the runtime, syntax, and execution flow of Python programs.", ["Python is easy to read and write.", "Use indentation to control blocks of code."], ["print('Hello CareerAI')", "name = 'Alex'\nprint(name)"], ["What is a variable?", "How does Python handle indentation?"]),
			learning_topic("Variables and Data Types", "Store values using variables and understand the common Python data types.", ["Use Python built-ins like int, float, str, bool, list, dict.", "Choose types carefully for data operations."], ["age = 24\nname = 'Alex'\nactive = True"], ["What is the difference between a list and a tuple?", "How do you check a value type?"]),
			learning_topic("Operators and Conditions", "Use arithmetic, comparison, and logic operators to make decisions.", ["Comparison operators return True or False.", "Use if, elif, and else for branch logic."], ["if score >= 80:\n    print('Great')\nelse:\n    print('Keep learning')"], ["How do logical operators work?", "When is a nested condition useful?"] ),
			learning_topic("Loops and Functions", "Repeat actions and reuse logic through functions.", ["For loops work well on iterables.", "Functions reduce duplication and improve clarity."], ["for i in range(3):\n    print(i)\n\ndef greet(name):\n    return f'Hi {name}'"], ["Which loop is better for fixed iterations?", "Why are functions useful in real projects?"] ),
			learning_topic("Lists, Tuples, Sets, and Dictionaries", "Use Python collections for structured data.", ["Lists are ordered and mutable.", "Tuples are fixed values.", "Sets remove duplicates.", "Dictionaries store key-value data."], ["skills = ['Python', 'SQL']\nprofile = {'name': 'Alex', 'score': 88}"], ["When would you use a set instead of a list?", "How do dictionary keys work?"] ),
			learning_topic("Strings and File Handling", "Work with text data and persist information to files.", ["String methods help clean and transform data.", "Use open() with read/write modes."], ["name.strip().lower()\nwith open('notes.txt', 'w') as f:\n    f.write('CareerAI')"], ["How do you read a file line by line?", "What is the difference between read() and readline()?"] ),
			learning_topic("Exception Handling and Modules", "Write resilient code and reuse functionality through modules.", ["Use try/except to handle failures gracefully.", "Modules help organize code into reusable units."], ["try:\n    int('abc')\nexcept ValueError:\n    print('Invalid input')"], ["Why is exception handling important?", "How do modules improve project structure?"] ),
			learning_topic("OOP and Advanced Python", "Model real entities with classes and leverage advanced Python features.", ["Classes define structure and behavior.", "OOP helps organize larger projects."], ["class User:\n    def __init__(self, name):\n        self.name = name"], ["What is inheritance?", "Why is __init__ used?"] ),
			learning_topic("Libraries and Practical Projects", "Use common libraries and turn theory into working applications.", ["Libraries help solve repeated problems faster.", "Build small projects to reinforce concepts."], ["import pandas as pd\nprint(pd.DataFrame({'skill': ['Python']}))"], ["How do you choose a library for a task?", "What is the best first project for learning Python?"] ),
		],
		"projects": [
			learning_project("Calculator", "Beginner", ["Python"], "Create a command-line calculator for arithmetic operations.", ["Python", "Functions"], ["Create a CLI interface", "Add operations for +, -, *, /", "Validate input and prevent crashes"], "A working calculator that demonstrates function design and error handling."),
			learning_project("To-Do App", "Intermediate", ["Python", "File Handling"], "Build a to-do manager that stores tasks in a file.", ["Python", "Lists", "File Handling"], ["Design a list-based data model", "Add add/remove/complete actions", "Persist tasks to a local file"], "A reusable task manager that introduces storage and logic flow."),
		],
	},
	"DSA": {
		"description": "Learn problem-solving patterns using arrays, lists, trees, graphs, and algorithm design.",
		"difficulty": "Intermediate",
		"duration": "6 weeks",
		"icon": ":material/stacked_line_chart:",
		"topics": [
			learning_topic("Arrays and Strings", "Work with indexed collections and common string operations.", ["Arrays are contiguous and efficient for indexing.", "Strings are often processed with slicing and iteration."], ["arr = [1, 2, 3]\ntext = 'careerai'"], ["How do array indexes work?", "Which string operations are common in coding interviews?"] ),
			learning_topic("Linked Lists and Stack", "Model dynamic collections and last-in-first-out data structures.", ["Linked lists are helpful when the size is dynamic.", "Stacks support undo, parsing, and backtracking."], ["stack = ['a', 'b']\nstack.append('c')"], ["When would you choose a linked list?", "How is a stack different from a queue?"] ),
			learning_topic("Queue, Searching, and Sorting", "Handle data processing patterns and lookup algorithms.", ["Searching answers where a value is located.", "Sorting organizes data for further analysis."], ["sorted([5, 2, 8])\n[2, 5, 8]"], ["What is binary search?", "How does quicksort work?"] ),
			learning_topic("Recursion and Trees", "Use recursive thinking for branching problems and hierarchical data.", ["Recursion solves smaller subproblems.", "Trees model hierarchical relationships."], ["def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)"], ["What is a base case?", "Why are tree traversals important?"] ),
			learning_topic("Graphs and Hashing", "Model networks and optimize lookup with hash tables.", ["Graphs are used for routing, networks, and dependency maps.", "Hashing allows constant-time lookup on average."], ["mapping = {'key': 'value'}"], ["What is a graph adjacency list?", "How is hashing used in Python dictionaries?"] ),
			learning_topic("Dynamic Programming", "Break complex problems into overlapping subproblems.", ["DP stores previous solutions to avoid repeated work.", "Use it for optimization and counting problems."], ["dp[0] = 1\n# optimize repeated calculations"], ["What is the difference between recursion and DP?", "Why do DP problems often need memoization?"] ),
			learning_topic("Time Complexity and Space Complexity", "Measure how efficient an algorithm is using Big O.", ["Big O estimates growth as input size increases.", "Choose algorithms based on both speed and memory."], ["O(n), O(log n), O(n^2)"], ["What is the difference between O(n) and O(log n)?", "How do you compare sorting algorithms?"] ),
		],
		"projects": [
			learning_project("Path Finder", "Intermediate", ["Python", "DSA"], "Implement a shortest-path planner for a small graph.", ["Graphs", "Queue", "Heuristics"], ["Model nodes and edges", "Use BFS or Dijkstra", "Validate path output"], "A graph-based route solver showing algorithmic reasoning."),
		],
	},
	"SQL": {
		"description": "Learn relational databases, queries, joins, and practical data retrieval patterns.",
		"difficulty": "Beginner",
		"duration": "3 weeks",
		"icon": ":material/database:",
		"topics": [
			learning_topic("SQL Basics and Database Concepts", "Understand relational tables, rows, columns, and database design.", ["A database stores structured records.", "Relational DBs use tables and keys."], ["SELECT * FROM students;"], ["What is the difference between a database and a table?", "Why are primary keys important?"] ),
			learning_topic("SELECT, WHERE, ORDER BY", "Filter and sort result sets for meaningful analysis.", ["SELECT picks columns.", "WHERE filters rows.", "ORDER BY sorts output."], ["SELECT name, score FROM students WHERE score > 80 ORDER BY score DESC;"], ["How do comparison operators work in SQL?", "How can ORDER BY improve reporting?"] ),
			learning_topic("GROUP BY, HAVING, Aggregate Functions", "Summarize records using counts, averages, and grouped results.", ["Aggregate functions reduce many rows to a summary.", "HAVING filters grouped output."], ["SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 50000;"], ["When should you use COUNT versus SUM?", "Why is HAVING used after grouping?"] ),
			learning_topic("Joins and Subqueries", "Combine related tables and nested queries for deeper analysis.", ["JOINs connect data across tables.", "Subqueries can help with filtering and derived tables."], ["SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;"], ["What is the difference between INNER JOIN and LEFT JOIN?", "When is a subquery useful?"] ),
			learning_topic("Constraints, Keys, and Normalization", "Ensure data quality and keep relational tables consistent.", ["Primary keys uniquely identify rows.", "Foreign keys link tables.", "Normalization reduces redundancy."], ["CREATE TABLE users (id INT PRIMARY KEY, name TEXT);"], ["Why use constraints?", "What is normalization?"] ),
			learning_topic("Views, Indexes, and Transactions", "Improve performance and maintain database consistency.", ["Indexes speed up queries.", "Transactions ensure atomic updates."], ["CREATE VIEW active_users AS SELECT * FROM users WHERE is_active = 1;"], ["When should you use a view instead of a table?", "What happens when a transaction fails?"] ),
		],
		"projects": [
			learning_project("Student Database", "Beginner", ["SQL"], "Use SQL to model and query a student management system.", ["SQL", "Database Design"], ["Create tables for students, courses, and enrollments", "Write queries for counts and top performers", "Add constraints and sample inserts"], "A simple relational database that demonstrates practical query design."),
		],
	},
	"HTML": {
		"description": "Master the structure of web documents and build accessible pages from the ground up.",
		"difficulty": "Beginner",
		"duration": "2 weeks",
		"icon": ":material/html:",
		"topics": [
			learning_topic("HTML Basics and Structure", "Understand document structure, metadata, and the browser display model.", ["The HTML document has html, head, and body sections.", "Semantic structure improves accessibility."], ["<html><body><h1>Hello</h1></body></html>"], ["What does the head section contain?", "Why is the body element important?"] ),
			learning_topic("Tags, Forms, and Tables", "Create content blocks, user inputs, and tabular layouts.", ["Forms collect user input.", "Tables are useful for structured data presentation."], ["<form><input type='text'></form>"], ["How do forms submit data?", "When is a table better than CSS grid?"] ),
			learning_topic("Links, Images, and Media", "Connect pages and embed content for richer experiences.", ["Images, videos, and links make pages informative and interactive."], ["<a href='https://careerai.example'>Learn more</a>"], ["What is the alt attribute for?", "How do you make an image user-friendly?"] ),
			learning_topic("Semantic HTML and HTML5", "Use modern HTML tags to improve structure and accessibility.", ["Semantic tags describe meaning.", "HTML5 adds many useful elements."], ["<header>, <nav>, <main>, <section>, <footer>"], ["Why is <main> useful?", "How does semantic markup help assistive technology?"] ),
		],
		"projects": [
			learning_project("Portfolio Landing Page", "Beginner", ["HTML", "CSS"], "Build a single-page portfolio with sections for about, work, and contact.", ["HTML", "Accessibility"], ["Add a hero section", "Create navigation links", "Structure sections semantically"], "A clean portfolio website that demonstrates page organization and content hierarchy."),
		],
	},
	"CSS": {
		"description": "Design visually polished interfaces using layout, color, typography, and responsive behavior.",
		"difficulty": "Beginner",
		"duration": "3 weeks",
		"icon": ":material/style:",
		"topics": [
			learning_topic("CSS Basics and Selectors", "Apply styles and target elements using selectors.", ["Selectors choose which elements receive style.", "Use external CSS for maintainability."], ["h1 { color: teal; }\n.card { padding: 16px; }"], ["What is the difference between class and id selectors?", "How does specificity work?"] ),
			learning_topic("Colors, Fonts, and Typography", "Improve readability and visual hierarchy.", ["Color and typography strongly influence user experience.", "Use consistent spacing and line-height."], ["body { font-family: Arial, sans-serif; }"], ["How do you choose good contrast?", "Why does font pairing matter?"] ),
			learning_topic("Box Model and Positioning", "Control layout with padding, borders, margin, and positioning.", ["Every element sits in a box.", "Positioning changes stacking and alignment behavior."], ["div { margin: 12px; padding: 20px; }"], ["What is the difference between relative and absolute positioning?", "When should you use display: inline-block?" ] ),
			learning_topic("Flexbox and Grid", "Create modern layout systems for responsive websites.", ["Flexbox is ideal for one-dimensional layouts.", "Grid handles rows and columns elegantly."], [".layout { display: flex; gap: 16px; }"], ["When do you prefer grid over flexbox?", "How do you center items with flexbox?"] ),
			learning_topic("Responsive Design and Animations", "Adapt designs across devices and add motion carefully.", ["Responsive design uses media queries.", "Animations improve feedback but should remain subtle."], ["@media (max-width: 768px) { .nav { display: block; } }"], ["What is a mobile-first design approach?", "How can animations increase usability?"] ),
		],
		"projects": [
			learning_project("Responsive Portfolio", "Intermediate", ["HTML", "CSS"], "Create a polished portfolio responsive across desktop and mobile screens.", ["CSS", "Responsive Design", "Flexbox"], ["Structure the page with sections", "Use flexbox and media queries", "Add hover states and polished spacing"], "A modern, responsive portfolio page with a consistent visual system."),
		],
	},
	"JavaScript": {
		"description": "Add interactivity, DOM updates, and async behavior to front-end applications.",
		"difficulty": "Intermediate",
		"duration": "4 weeks",
		"icon": ":material/javascript:",
		"topics": [
			learning_topic("JavaScript Basics", "Learn core syntax, execution, and browser integration.", ["JavaScript runs in browsers and on servers.", "Use variables and functions to build logic."], ["const age = 21;\nconsole.log(age);"], ["What is the difference between var, let, and const?", "Why is JavaScript essential for the web?"] ),
			learning_topic("Variables, Data Types, and Operators", "Work with primitive and reference types in JavaScript.", ["Numbers, strings, booleans, arrays, and objects are core data types.", "Operators help with comparisons and math."], ["const total = 10 + 5;\nconst ready = true;"], ["How do objects differ from arrays?", "What is strict equality?"] ),
			learning_topic("Conditions, Loops, and Functions", "Control program flow and build reusable logic.", ["Functions help organize code.", "Loops handle repeated tasks efficiently."], ["for (let i = 0; i < 3; i++) { console.log(i); }"], ["When do you use for vs while loops?", "Why are functions important in DOM-driven code?"] ),
			learning_topic("Arrays, Objects, and DOM", "Store data and manipulate page elements.", ["Objects represent structured data.", "DOM APIs let you update page content and styles."], ["document.getElementById('title').textContent = 'Hello';"], ["How do you select elements in the DOM?", "What is the role of objects in UI logic?"] ),
			learning_topic("Events and Asynchronous JavaScript", "Handle user actions and non-blocking logic.", ["Event listeners react to clicks and form submissions.", "Async operations use promises and async/await."], ["button.addEventListener('click', () => console.log('Clicked'));"], ["What is async/await?", "How do promises help with HTTP calls?"] ),
			learning_topic("APIs" , "Connect applications to external data sources and services.", ["Use fetch() to request data from APIs.", "APIs return structured JSON data."], ["fetch('/api/data').then(res => res.json());"], ["What is JSON?", "How do you handle API errors?"] ),
		],
		"projects": [
			learning_project("Interactive To-Do App", "Intermediate", ["JavaScript", "HTML", "CSS"], "Build a complete to-do web app with add, delete, and completion interactions.", ["JavaScript", "DOM", "Events"], ["Add event listeners", "Render tasks dynamically", "Update status and counts"], "A functional front-end app that shows real UI state management."),
		],
	},
	"FastAPI": {
		"description": "Build APIs with FastAPI and prepare AI/ML services for real deployment.",
		"difficulty": "Advanced",
		"duration": "4 weeks",
		"icon": ":material/api:",
		"topics": [
			learning_topic("FastAPI Introduction", "Understand the philosophy and strengths of FastAPI as a modern API framework.", ["FastAPI is fast, easy to document, and strongly typed.", "It is ideal for AI and backend services."], ["from fastapi import FastAPI\napp = FastAPI()"], ["Why is FastAPI useful for AI services?", "What makes FastAPI easier to document than plain Flask?"] ),
			learning_topic("API Concepts and HTTP Methods", "Learn the request/response lifecycle and core HTTP verbs.", ["GET reads data, POST creates data, PUT updates, DELETE removes.", "REST APIs are resource-oriented."], ["@app.get('/users')\nasync def list_users():\n    return []"], ["What is the difference between GET and POST?", "Why are HTTP methods important for REST APIs?"] ),
			learning_topic("Routes, Parameters, and Validation", "Create endpoints that accept path, query, and payload data safely.", ["Validation reduces runtime edge cases.", "Path and query parameters guide API behavior."], ["@app.get('/predict/{user_id}')"], ["How do you validate incoming request data?", "What is the purpose of a path parameter?"] ),
			learning_topic("Pydantic and Models", "Use Python data models to validate and structure API payloads.", ["Pydantic allows structured validation.", "Models improve readability and reduce bugs."], ["from pydantic import BaseModel\nclass UserInput(BaseModel):\n    name: str"], ["Why use BaseModel?", "How does validation protect the API?"] ),
			learning_topic("CRUD APIs and Database Integration", "Connect API endpoints to persistent data storage.", ["APIs often manage create, read, update, delete flows.", "Use SQL or ORM layers for persistence."], ["@app.post('/jobs')\nasync def add_job():"], ["How do CRUD operations map to HTTP methods?", "When is a database layer necessary?"] ),
			learning_topic("Authentication and API Documentation", "Protect endpoints and expose clear documentation to users and teams.", ["Authentication ensures only authorized users access protected routes.", "FastAPI automatically generates OpenAPI docs."], ["app = FastAPI(title='CareerAI API')"], ["Why is API documentation important?", "How would you secure a production API?"] ),
		],
		"projects": [
			learning_project("ML Model API", "Advanced", ["FastAPI", "Python", "Pydantic"], "Create a production-style API for a machine learning prediction endpoint.", ["FastAPI", "Pydantic", "Model Deployment"], ["Create routes for input and output", "Use validation models", "Return JSON predictions and handle errors"], "A deployable prediction API aligned with real AI product workflows."),
		],
	},
	"Machine Learning": {
		"description": "Learn predictive modeling, evaluation, and the end-to-end ML lifecycle.",
		"difficulty": "Intermediate",
		"duration": "6 weeks",
		"icon": ":material/precision_manufacturing:",
		"topics": [
			learning_topic("Introduction to ML and Types of ML", "Understand supervised, unsupervised, and reinforcement learning.", ["ML learns patterns from data.", "Different tasks require different labels and problem structures."], ["Supervised learning: labeled data\nUnsupervised learning: hidden patterns"], ["What is the difference between supervised and unsupervised learning?", "When is classification used?"] ),
			learning_topic("Regression and Classification", "Predict continuous and categorical outcomes.", ["Regression predicts numbers.", "Classification predicts groups or classes."], ["y = a * x + b\nclassify spam vs not spam"], ["Which metric is best for a regression problem?", "What is a confusion matrix?"] ),
			learning_topic("Clustering and Feature Engineering", "Discover patterns in unlabeled data and improve model quality.", ["Feature engineering changes raw data into useful signals.", "Clustering groups similar data points."], ["k-means on customer segments"], ["Why is feature engineering important?", "What is the role of scaling in ML?"] ),
			learning_topic("Data Preprocessing and Train/Test Split", "Prepare data for modeling and avoid overfitting leaks.", ["Cleaning ensures consistent values.", "Train/test split measures generalization."], ["from sklearn.model_selection import train_test_split"], ["Why is data cleaning important?", "What is target leakage?"] ),
			learning_topic("Model Evaluation and Cross Validation", "Measure performance and validate generalization robustly.", ["Accuracy alone can be misleading.", "Cross-validation helps with reliable comparison."], ["from sklearn.model_selection import cross_val_score"], ["What is precision vs recall?", "How does cross-validation help?"] ),
			learning_topic("Overfitting, Underfitting, and Hyperparameter Tuning", "Tune models for the right balance between complexity and generalization.", ["Overfitting memorizes training data.", "Underfitting is too simple for the pattern."], ["GridSearchCV for tuning"], ["What is bias-variance tradeoff?", "How can you detect overfitting?"] ),
			learning_topic("Scikit-learn and ML Projects", "Build practical projects with real-world workflows.", ["Scikit-learn provides a consistent ML workflow.", "Projects help connect theory to product tasks."], ["from sklearn.linear_model import LinearRegression"], ["How do pipelines simplify ML workflows?", "Why are project portfolios valuable?"] ),
		],
		"projects": [
			learning_project("House Price Prediction", "Intermediate", ["Python", "Scikit-learn", "Pandas"], "Use tabular data to predict purchase price for properties.", ["Machine Learning", "Regression", "Feature Engineering"], ["Prepare the dataset", "Train and evaluate a model", "Compare multiple algorithms"], "A complete ML workflow from data cleaning to model evaluation."),
		],
	},
	"AI": {
		"description": "Build AI systems that combine reasoning, learning, and product value using modern tools.",
		"difficulty": "Intermediate",
		"duration": "5 weeks",
		"icon": ":material/psychology:",
		"topics": [
			learning_topic("Introduction to AI and Intelligent Systems", "Understand the history and patterns behind AI systems.", ["AI helps machines perform tasks that usually require human intelligence.", "Intelligent systems combine logic, perception, and adaptation."], ["AI use cases: recommendation, classification, planning"], ["What is AI versus traditional software?", "What are common AI applications?"] ),
			learning_topic("Machine Learning vs AI and Deep Learning", "See how ML and deep learning are part of the broader AI landscape.", ["ML is one approach within AI.", "Deep learning uses neural networks to learn complex patterns."], ["AI -> ML -> Deep Learning"], ["How do classical approaches differ from deep learning?", "When is deep learning the right tool?" ] ),
			learning_topic("Neural Networks", "Model layered computation for pattern recognition and representation learning.", ["Neural networks consist of layers and activations.", "They can learn from large amounts of data."], ["Input -> Hidden -> Output"], ["What is a neuron?", "Why are hidden layers important?"] ),
			learning_topic("Generative AI and LLM Concepts", "Understand text generation, prompts, and transformer-based models.", ["Generative models create output from patterns.", "LLMs are trained on massive corpora of language data."], ["Prompt -> Model -> Response"], ["What is prompting?", "Why do transformer models matter for LLMs?"] ),
			learning_topic("AI Applications", "Apply AI practically to product and operations workflows.", ["AI improves search, support, personalization, and analysis.", "Business use cases should be measurable."], ["AI chatbot for onboarding"], ["How can AI increase product value?", "What makes an AI product trustworthy?"] ),
		],
		"projects": [
			learning_project("AI Chatbot", "Advanced", ["Python", "NLP", "FastAPI"], "Create a conversational assistant for a simple support or learning scenario.", ["AI", "NLP", "Prompting"], ["Design the conversation flow", "Connect to a model or rule engine", "Test interactive responses"], "A working AI chatbot prototype demonstrating practical application design."),
		],
	},
	"NLP": {
		"description": "Process language data for sentiment analysis, extraction, and real-world text intelligence.",
		"difficulty": "Advanced",
		"duration": "4 weeks",
		"icon": ":material/text_fields:",
		"topics": [
			learning_topic("NLP Introduction and Text Preprocessing", "Prepare raw text for analysis by cleaning and standardizing it.", ["Tokenization, lowercasing, and formatting are essential first steps.", "Clean input improves downstream model performance."], ["text.lower().replace(',', ' ')"], ["Why is text preprocessing necessary?", "What are common preprocessing problems?"] ),
			learning_topic("Tokenization, Stop Words, and Stemming", "Break text into units and reduce noise.", ["Tokenization splits words or sentences.", "Stop words are common but not always semantically useful."], ["['career', 'ai', 'learning']"], ["How is stemming different from lemmatization?", "When would you remove stop words?"] ),
			learning_topic("Lemmatization and POS Tagging", "Reduce words to valid dictionary forms and understand grammatical roles.", ["Lemmatization keeps meaning while normalizing forms.", "POS tagging helps interpret structure."], ["'running' -> 'run'"], ["Why is lemmatization important for analysis?", "How does POS help in text applications?"] ),
			learning_topic("Named Entity Recognition and Text Classification", "Identify people, organizations, and categories in text.", ["NER finds entities.", "Text classification assigns labels to documents or sentences."], ["classify customer feedback as positive/negative"], ["How is NER used in resume parsing?", "What is a text classifier?" ] ),
			learning_topic("Sentiment Analysis, TF-IDF, and Embeddings", "Represent text numerically and detect sentiment or relevance.", ["TF-IDF emphasizes important words.", "Embeddings capture semantic relationships."], ["from sklearn.feature_extraction.text import TfidfVectorizer"], ["When is TF-IDF enough?", "What are embeddings used for?" ] ),
		],
		"projects": [
			learning_project("Sentiment Analyzer", "Advanced", ["Python", "NLP", "Scikit-learn"], "Build a sentiment classifier for product reviews or social media texts.", ["Text Preprocessing", "Classification", "Evaluation"], ["Clean the input text", "Vectorize features", "Train and evaluate a model"], "A working sentiment-analysis pipeline with measurable performance."),
		],
	},
	"OpenCV": {
		"description": "Learn computer vision pipelines for image and video processing tasks.",
		"difficulty": "Intermediate",
		"duration": "3 weeks",
		"icon": ":material/image_search:",
		"topics": [
			learning_topic("OpenCV Introduction", "Understand the role of OpenCV in image analysis and visual AI.", ["OpenCV is a library for vision tasks.", "It helps with image capture, processing, and analysis."], ["import cv2"], ["What is computer vision?", "How is OpenCV used in real products?"] ),
			learning_topic("Image Reading and Processing", "Load and manipulate images in Python.", ["OpenCV reads images as NumPy arrays.", "Image processing often starts with resizing and normalization."], ["img = cv2.imread('photo.jpg')"], ["What does a color image array represent?", "Why is resizing useful?"] ),
			learning_topic("Image Filtering and Edge Detection", "Enhance or detect important visual features.", ["Filtering reduces noise or emphasizes patterns.", "Edge detection helps identify boundaries."], ["cv2.Canny(img, 100, 200)"], ["What is a Gaussian blur used for?", "How does the Canny algorithm help?"] ),
			learning_topic("Object Detection Concepts and Video Processing", "Understand detection pipelines and motion-based analysis.", ["Object detection identifies regions of interest.", "Video processing involves frame-by-frame analysis."], ["for frame in video: ..."], ["What is a frame in video processing?", "How are computer vision models deployed in practice?"] ),
		],
		"projects": [
			learning_project("Image Filter Studio", "Intermediate", ["Python", "OpenCV"], "Create a small image-processing app with filters and edge detection.", ["OpenCV", "Image Processing", "Visualization"], ["Read an image", "Apply filters and edge detection", "Display the results"], "A computer-vision prototype that demonstrates image manipulation workflows."),
		],
	},
	"Git": {
		"description": "Understand version control basics and daily collaboration workflows with Git.",
		"difficulty": "Beginner",
		"duration": "2 weeks",
		"icon": ":material/commit:",
		"topics": [
			learning_topic("Git Introduction", "Track code changes and collaborate safely.", ["Git captures snapshots of your project.", "It helps you review history and recover quickly."], ["git init"], ["What problem does Git solve?", "Why is history important in software development?"] ),
			learning_topic("Repository Basics", "Create, inspect, and manage repositories.", ["A repository is a project folder with Git metadata.", "Use status to inspect changes."], ["git status"], ["What is the difference between tracked and untracked files?", "Why does git status matter?"] ),
			learning_topic("Add, Commit, Push, and Pull", "Record and synchronize project changes.", ["Commit saves changes locally.", "Push sends them to remote servers."], ["git add .\ngit commit -m 'Initial version'\ngit push"], ["What does git add do?", "When should you create a commit?"] ),
			learning_topic("Branching, Merge, and Conflict Handling", "Work independently and handle changes carefully.", ["Branches isolate work streams.", "Merges combine work, and conflicts may need manual decisions."], ["git checkout -b feature/login"], ["How do branches support team collaboration?", "What is a merge conflict?"] ),
		],
		"projects": [
			learning_project("Git Workflow Practice", "Beginner", ["Git"], "Use Git to create a small project branch, commit changes, and merge a feature branch.", ["Git", "Collaboration"], ["Initialize a repo", "Create a branch", "Commit and merge the branch"], "A clean Git workflow demonstration with version control confidence."),
		],
	},
	"GitHub": {
		"description": "Use GitHub to host repositories, document work, and build a strong portfolio.",
		"difficulty": "Beginner",
		"duration": "2 weeks",
		"icon": ":material/account_tree:",
		"topics": [
			learning_topic("Repository Creation and README", "Publish projects with clear structure and documentation.", ["A README helps visitors understand the project.", "Good documentation improves adoption."], ["README.md"], ["Why do projects need a README?", "What should a good README include?"] ),
			learning_topic("Branches, Pull Requests, and Issues", "Collaborate through review and structured work flows.", ["Pull requests support code review.", "Issues track tasks and bugs."], ["Create PR from feature branch to main"], ["How do pull requests improve quality?", "When should you open an issue?"] ),
			learning_topic("Releases and GitHub Portfolio", "Present projects professionally and maintain release history.", ["Releases show milestone snapshots.", "A GitHub portfolio helps recruiters evaluate work."], ["Create release tag and notes"], ["How can GitHub strengthen job applications?", "Why are tags and releases useful?"] ),
		],
		"projects": [
			learning_project("Portfolio Repository", "Beginner", ["GitHub", "Git"], "Build a polished GitHub repository for a portfolio project.", ["GitHub", "Documentation"], ["Write a project README", "Add project sections and screenshots", "Create a structured repo"], "A professional GitHub profile piece showcasing technical execution."),
		],
	},
	"Docker": {
		"description": "Package applications into containers and standardize deployment environments.",
		"difficulty": "Intermediate",
		"duration": "3 weeks",
		"icon": ":material/container:",
		"topics": [
			learning_topic("Docker Introduction", "Understand how containers package app code and dependencies.", ["Containers are lightweight and portable.", "They make environments consistent across machines."], ["docker run python:3.11"], ["Why is containerization useful?", "How is Docker different from virtualization?"] ),
			learning_topic("Images, Containers, and Dockerfile", "Build and run application environments using Docker files.", ["Dockerfiles define application setup steps.", "Images are reusable blueprints for containers."], ["FROM python:3.11\nRUN pip install fastapi"], ["What is the role of a Dockerfile?", "How do images relate to containers?"] ),
			learning_topic("Ports, Volumes, and Environment Variables", "Control access and persist configuration in containers.", ["Ports map app traffic inside and outside a container.", "Volumes keep data outside the container layer."], ["docker run -p 8000:8000"], ["Why are volumes important for data?", "How do environment variables improve config management?"] ),
			learning_topic("Docker Compose and Real Applications", "Coordinate multi-service applications using Docker Compose.", ["Compose helps run applications with dependencies.", "It simplifies local development and testing."], ["docker-compose up"], ["When do you need multiple containers together?", "How does Compose help app onboarding?"] ),
		],
		"projects": [
			learning_project("Containerize a Python API", "Intermediate", ["Docker", "FastAPI"], "Package a Python API into a container and run it with environment configuration.", ["Docker", "FastAPI", "Environment Variables"], ["Write a Dockerfile", "Expose a port", "Run the app inside a container"], "A reproducible deployment unit for a backend service."),
		],
	},
	"AWS": {
		"description": "Learn the fundamentals of AWS and how cloud services support business and AI workloads.",
		"difficulty": "Intermediate",
		"duration": "5 weeks",
		"icon": ":material/cloud:",
		"topics": [
			learning_topic("Cloud Computing Basics and AWS Introduction", "Understand cloud concepts, shared responsibility, and the value of managed services.", ["Cloud lets you deploy resources on demand.", "AWS offers compute, storage, and networking services."], ["EC2, S3, RDS"], ["What is cloud computing?", "Why do companies move workloads to AWS?"] ),
			learning_topic("IAM, EC2, and S3", "Secure access and deploy applications and data in the cloud.", ["IAM controls identity and permissions.", "EC2 hosts virtual servers and S3 stores objects."], ["aws s3 ls"], ["Why is IAM critical for security?", "When is S3 the right service?"] ),
			learning_topic("RDS, Deployment, and Hosting", "Run databases and host applications in managed services.", ["Managed services reduce operational effort.", "Hosting focuses on deployment and uptime."], ["Deploy to EC2 or container platform"], ["How does RDS simplify database operations?", "What is the difference between hosting and deployment?"] ),
			learning_topic("AI/ML Deployment Concepts", "Prepare for cloud-based model hosting and scalable inference.", ["AI applications may need storage, autoscaling, and secure APIs.", "Cloud services simplify deployment."], ["Model endpoint + API service"], ["How is ML deployment different from traditional hosting?", "What does inference mean in AI systems?"] ),
		],
		"projects": [
			learning_project("Deploy a Small API on AWS", "Advanced", ["AWS", "Docker", "FastAPI"], "Host a simple application or API on a cloud service with managed resources.", ["AWS", "Deployment", "Security"], ["Set up an EC2 or lightweight cloud deployment", "Expose the service", "Document the setup"], "A working cloud deployment setup that demonstrates application hosting fundamentals."),
		],
	},
	"Projects": {
		"description": "Turn your learning into portfolio-ready projects connected to real products and workflows.",
		"difficulty": "All levels",
		"duration": "Flexible",
		"icon": ":material/folder:",
		"topics": [
			learning_topic("Project Planning", "Build a clear plan before you code.", ["Good projects begin with a problem definition.", "Break work into manageable phases."], ["Define problem, dataset, features, deliverable"], ["Why do we plan before implementation?", "What should a project brief include?"] ),
			learning_topic("Build in Small Iterations", "Ship a working version early and improve it.", ["Iterative delivery reduces risk.", "Small milestones improve confidence and learning."], ["Prototype -> validate -> iterate"], ["Why is iteration better than one massive final push?", "How does feedback improve project quality?"] ),
			learning_topic("Documentation and Portfolio Readiness", "Present the project clearly for recruiters and teams.", ["Clear docs help users and reviewers understand your work.", "A portfolio project is a proof of ability."], ["README, architecture, screenshots"], ["Why is documentation a core part of project quality?", "What makes a project recruiter-friendly?"] ),
		],
		"projects": [
			learning_project("Career Recommendation API", "Advanced", ["Python", "FastAPI", "SQL"], "Build an API that recommends relevant career paths based on user profiles or skill data.", ["FastAPI", "SQL", "Career Guidance"], ["Model the data inputs", "Build recommendation logic", "Expose an endpoint and document usage"], "A project that combines career technology with backend service design."),
			learning_project("Resume Analyzer", "Advanced", ["Python", "NLP", "Machine Learning"], "Parse and evaluate resume content to provide ATS-style insights.", ["Resume analysis", "NLP", "Python"], ["Extract text from a PDF", "Score the resume", "Return keywords and suggestions"], "A product-oriented AI workflow relevant to CareerAI-style use cases."),
		],
	},
	"Progress Tracking": {
		"description": "Measure how consistently you learn, which topics are complete, and how close you are to the next milestone.",
		"difficulty": "Beginner",
		"duration": "Flexible",
		"icon": ":material/trending_up:",
		"topics": [
			learning_topic("Course Progress", "Track milestone completion for each course.", ["Progress helps maintain momentum.", "Measure completion from small wins."], ["50% progress in Python course"], ["How do you define a completed course?", "Why is consistency more useful than intensity?"] ),
			learning_topic("Topic Completion and Time", "Evaluate how much of a course is complete and how long it takes.", ["Completion can be tracked topic by topic.", "Time-on-task is a useful self-metric."], ["Python Basics 65%"], ["How can you use progress data to plan next steps?", "Why is time tracking useful in learning?"] ),
			learning_topic("Learning Streaks and Momentum", "Stay motivated with consistent learning cycles.", ["Streaks maintain habit formation.", "Momentum makes learning easier over time."], ["3-day streak"], ["What helps maintain a learning streak?", "How do small wins compound over time?"] ),
		],
		"projects": [
			learning_project("Learning Dashboard", "Beginner", ["Python", "Streamlit", "Pandas"], "Create a light dashboard that shows categories, topics, and course progress.", ["Progress tracking", "Visualization", "Dashboard design"], ["Model the learning records", "Display progress bars", "Summarize completion states"], "A progress dashboard suitable for personal and platform-level learning insights."),
		],
	},
}


def seed_learning_hub_data() -> None:
	with connection() as database:
		for category_name, settings in LEARNING_HUB_COURSE_DATA.items():
			slug = re.sub(r"[^a-z0-9]+", "-", category_name.lower()).strip('-') or "learning"
			category_row = database.execute(
				"SELECT id FROM learning_categories WHERE slug = ?",
				(slug,),
			).fetchone()
			if category_row is None:
				cursor = database.execute(
					"INSERT INTO learning_categories (slug, name, description, icon, difficulty, duration) VALUES (?, ?, ?, ?, ?, ?)",
					(slug, category_name, settings["description"], settings["icon"], settings["difficulty"], settings["duration"]),
				)
				category_id = int(cursor.lastrowid)
			else:
				category_id = int(category_row["id"])
			course_name = category_name
			course_row = database.execute(
				"SELECT id FROM learning_courses WHERE category_id = ? AND course_name = ?",
				(category_id, course_name),
			).fetchone()
			if course_row is None:
				course_cursor = database.execute(
					"INSERT INTO learning_courses (category_id, course_name, description, difficulty, duration, icon, tags_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
					(category_id, course_name, settings["description"], settings["difficulty"], settings["duration"], settings["icon"], json.dumps([category_name, settings["difficulty"]])),
				)
				course_id = int(course_cursor.lastrowid)
			else:
				course_id = int(course_row["id"])
			for topic in settings.get("topics", []):
				topic_row = database.execute(
					"SELECT id FROM learning_topics WHERE course_id = ? AND topic_name = ?",
					(course_id, topic["title"]),
				).fetchone()
				if topic_row is None:
					topic_cursor = database.execute(
						"INSERT INTO learning_topics (course_id, topic_name, explanation, notes_json, examples_json, questions_json, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
						(course_id, topic["title"], topic["explanation"], json.dumps(topic["notes"]), json.dumps(topic["examples"]), json.dumps(topic["questions"]), topic["status"]),
					)
					topic_id = int(topic_cursor.lastrowid)
				else:
					topic_id = int(topic_row["id"])
				for index, lesson_name in enumerate(["Core concepts", "Notes and examples", "Practice guide"], start=1):
					lesson_text = f"Lesson {index} for {topic['title']}: {topic['explanation']}"
					database.execute(
						"INSERT OR IGNORE INTO learning_lessons (topic_id, lesson_title, lesson_content) VALUES (?, ?, ?)",
						(topic_id, lesson_name, lesson_text),
					)
			for project in settings.get("projects", []):
				project_row = database.execute(
					"SELECT id FROM learning_projects WHERE course_id = ? AND project_title = ?",
					(course_id, project["title"]),
				).fetchone()
				if project_row is None:
					database.execute(
						"INSERT INTO learning_projects (course_id, project_title, difficulty, technologies_json, objective, required_skills_json, project_steps_json, expected_outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
						(course_id, project["title"], project["difficulty"], json.dumps(project["technologies"]), project["objective"], json.dumps(project["required_skills"]), json.dumps(project["steps"]), project["expected_outcome"]),
					)


def list_learning_categories() -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute(
			"SELECT id, slug, name, description, icon, difficulty, duration FROM learning_categories ORDER BY name"
		).fetchall()
		return [dict(row) for row in rows]


def list_learning_courses(category_name: str | None = None) -> list[dict[str, Any]]:
	with connection() as database:
		if category_name:
			rows = database.execute(
				"SELECT c.id, c.course_name, c.description, c.difficulty, c.duration, c.icon, c.tags_json, cat.name AS category_name FROM learning_courses c JOIN learning_categories cat ON cat.id = c.category_id WHERE cat.name = ? AND c.active = 1 ORDER BY c.course_name",
				(category_name,),
			).fetchall()
		else:
			rows = database.execute(
				"SELECT c.id, c.course_name, c.description, c.difficulty, c.duration, c.icon, c.tags_json, cat.name AS category_name FROM learning_courses c JOIN learning_categories cat ON cat.id = c.category_id WHERE c.active = 1 ORDER BY cat.name, c.course_name"
			).fetchall()
		return [dict(row) for row in rows]


def get_learning_course_details(course_name: str) -> dict[str, Any] | None:
	with connection() as database:
		course = database.execute(
			"SELECT c.id, c.course_name, c.description, c.difficulty, c.duration, c.icon, c.tags_json, cat.name AS category_name FROM learning_courses c JOIN learning_categories cat ON cat.id = c.category_id WHERE c.course_name = ? AND c.active = 1 LIMIT 1",
			(course_name,),
		).fetchone()
		if not course:
			return None
		course_data = dict(course)
		course_data["topics"] = [dict(item) for item in database.execute(
			"SELECT id, topic_name AS title, explanation, notes_json, examples_json, questions_json, status FROM learning_topics WHERE course_id = ? ORDER BY id",
			(course["id"],),
		).fetchall()]
		for topic in course_data["topics"]:
			topic["notes"] = json.loads(topic.pop("notes_json") or '[]')
			topic["examples"] = json.loads(topic.pop("examples_json") or '[]')
			topic["questions"] = json.loads(topic.pop("questions_json") or '[]')
			lesson_rows = database.execute(
				"SELECT lesson_title, lesson_content FROM learning_lessons WHERE topic_id = ? ORDER BY id",
				(topic["id"],),
			).fetchall()
			topic["lessons"] = [dict(item) for item in lesson_rows]
		course_data["projects"] = [dict(item) for item in database.execute(
			"SELECT project_title AS title, difficulty, technologies_json, objective, required_skills_json, project_steps_json, expected_outcome FROM learning_projects WHERE course_id = ? ORDER BY id",
			(course["id"],),
		).fetchall()]
		for project in course_data["projects"]:
			project["technologies"] = json.loads(project.pop("technologies_json") or '[]')
			project["required_skills"] = json.loads(project.pop("required_skills_json") or '[]')
			project["steps"] = json.loads(project.pop("project_steps_json") or '[]')
		return course_data


def ensure_learning_user_progress(user_id: int) -> None:
	with connection() as database:
		course_rows = database.execute("SELECT id, course_name FROM learning_courses").fetchall()
		for course in course_rows:
			database.execute(
				"INSERT OR IGNORE INTO learning_user_course_progress (user_id, course_id, progress_percent, last_topic, status) VALUES (?, ?, 0, '', 'Not started')",
				(user_id, course["id"]),
			)
			topic_rows = database.execute("SELECT id, topic_name FROM learning_topics WHERE course_id = ?", (course["id"],)).fetchall()
			for topic in topic_rows:
				database.execute(
					"INSERT OR IGNORE INTO learning_user_topic_progress (user_id, topic_id, progress_percent, status) VALUES (?, ?, 0, 'Not started')",
					(user_id, topic["id"]),
				)
		project_rows = database.execute("SELECT id, project_title FROM learning_projects WHERE course_id = ?", (course["id"],)).fetchall()
		for project in project_rows:
			database.execute(
				"INSERT OR IGNORE INTO learning_user_project_progress (user_id, project_id, progress_percent, status) VALUES (?, ?, 0, 'Not started')",
				(user_id, project["id"]),
			)


def update_topic_progress(user_id: int, course_name: str, topic_title: str, progress_value: int, status: str | None = None) -> None:
	with connection() as database:
		course_id = database.execute("SELECT id FROM learning_courses WHERE course_name = ?", (course_name,)).fetchone()
		if not course_id:
			return
		topic = database.execute(
			"SELECT id FROM learning_topics WHERE course_id = ? AND topic_name = ?",
			(course_id["id"], topic_title),
		).fetchone()
		if not topic:
			return
		completion = max(0, min(100, int(progress_value)))
		next_status = status or ("Completed" if completion >= 100 else "In progress" if completion > 0 else "Not started")
		database.execute(
			"UPDATE learning_user_topic_progress SET progress_percent = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND topic_id = ?",
			(completion, next_status, user_id, topic["id"]),
		)
		database.execute(
			"UPDATE learning_user_course_progress SET progress_percent = MAX(progress_percent, ?), last_topic = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND course_id = ?",
			(completion, topic_title, next_status, user_id, course_id["id"]),
		)
		database.execute(
			"INSERT INTO learning_activity (user_id, course_name, topic_name, progress_percent, activity_type) VALUES (?, ?, ?, ?, 'learning')",
			(user_id, course_name, topic_title, completion),
		)


def normalize_skill_name(skill: str) -> str:
	return re.sub(r"[\s_]+", " ", str(skill).strip().lower()).replace("-", " ")


def get_learning_recommendations_for_user(user_id: int) -> list[dict[str, Any]]:
	profile = get_user_profile(user_id)
	career_rows = get_recommendations_for_user(user_id)
	missing_words: list[str] = []
	for row in career_rows[:4]:
		missing_words.extend(row.get("missing_skills", []))
	if not missing_words:
		missing_words = ["Python", "SQL", "AI"]
	if profile.get("skills"):
		missing_words.extend(profile.get("skills", []))
	key_map = {
		"Python": ["python", "development", "backend", "automation"],
		"SQL": ["sql", "database", "query", "data analysis"],
		"Machine Learning": ["machine learning", "ml", "ai", "classification"],
		"AI": ["ai", "nlp", "prompting", "generative ai"],
		"NLP": ["nlp", "text", "language", "sentiment"],
		"FastAPI": ["fastapi", "api", "backend"],
		"Docker": ["docker", "container", "deployment"],
		"AWS": ["aws", "cloud", "hosting"],
		"Git": ["git", "github", "version control"],
	}
	prioritized: list[dict[str, Any]] = []
	seen: set[str] = set()
	for item in missing_words:
		needle = normalize_skill_name(str(item))
		for course_name, aliases in key_map.items():
			normalized_aliases = [normalize_skill_name(alias) for alias in aliases]
			if any(alias in needle for alias in normalized_aliases):
				if course_name not in seen:
					seen.add(course_name)
					prioritized.append({
						"course_name": course_name,
						"reason": "This applies to your current career direction and fills a recurring skill gap for real job readiness.",
						"priority": "High" if course_name in {"Python", "SQL", "Machine Learning", "FastAPI"} else "Medium",
					})
				break
	if not prioritized:
		prioritized = [
			{"course_name": "Python", "reason": "Strong foundation for automation, application logic, and product work.", "priority": "High"},
			{"course_name": "SQL", "reason": "Useful for data handling, reporting, and backend workflows.", "priority": "High"},
			{"course_name": "Machine Learning", "reason": "Ideal for building intelligent features and career growth in AI-driven roles.", "priority": "Medium"},
		]
	return prioritized

def get_user_skill_snapshot(user_id: int) -> set[str]:
	profile = get_user_profile(user_id)
	skills: list[str] = []
	skills.extend(profile.get("skills", []) or [])
	if profile.get("career_goal"):
		skills.append(profile["career_goal"])
	if profile.get("education"):
		skills.append(profile["education"])
	if profile.get("experience_summary"):
		skills.extend(re.findall(r"[A-Za-z][A-Za-z+.#/ -]{2,}", profile["experience_summary"]))
	with connection() as database:
		for row in database.execute(
			"SELECT DISTINCT skill FROM extracted_skills WHERE user_id = ? AND source IN ('resume', 'profile') ORDER BY skill",
			(user_id,),
		):
			skills.append(row["skill"])
	return {normalize_skill_name(skill) for skill in skills if skill and str(skill).strip()}


def career_match_score(user_skills: set[str], required_skills: list[str]) -> tuple[list[str], list[str], list[str], int]:
	normalized_required = [normalize_skill_name(skill) for skill in required_skills]
	exact_matches = [skill for skill in required_skills if normalize_skill_name(skill) in user_skills]
	missing = [skill for skill in required_skills if normalize_skill_name(skill) not in user_skills]
	partial = []
	for skill in required_skills:
		normalized = normalize_skill_name(skill)
		if normalized not in user_skills and any(word in user_skills for word in normalized.split() if len(word) > 2):
			partial.append(skill)
	if required_skills:
		gap = max(0, round((len(missing) / len(required_skills)) * 100))
	else:
		gap = 0
	return exact_matches, missing, partial, gap


def get_role_dataset(career_name: str) -> dict[str, Any]:
	for role_name, payload in CAREER_DATASET.items():
		if role_name.lower() == career_name.lower():
			return payload
	return {}


@lru_cache(maxsize=1)
def get_career_tfidf_model() -> tuple[list[str], TfidfVectorizer, Any]:
	role_names = list(CAREER_DATASET)
	role_documents = [
		" ".join(
			[
				name,
				payload["description"],
				" ".join(payload["required_skills"]),
				" ".join(payload.get("optional_skills", [])),
				" ".join(payload.get("related_technologies", [])),
				payload["learning_direction"],
				payload["overview"],
			]
		)
		for name, payload in CAREER_DATASET.items()
	]
	vectorizer = TfidfVectorizer(
		ngram_range=(1, 2),
		stop_words="english",
		sublinear_tf=True,
		token_pattern=r"(?u)\b\w[\w+#.-]*\b",
	)
	role_vectors = vectorizer.fit_transform(role_documents)
	return role_names, vectorizer, role_vectors


def compute_career_recommendations(user_id: int) -> list[dict[str, Any]]:
	profile = get_user_profile(user_id)
	analysis = get_analysis(user_id)
	user_skills = get_user_skill_snapshot(user_id)
	profile_values = [
		profile.get("career_goal", ""),
		profile.get("education", ""),
		profile.get("experience_summary", ""),
	]
	for key in ("skills", "interests"):
		values = profile.get(key) or []
		if isinstance(values, str):
			values = [item.strip() for item in values.split(",") if item.strip()]
		profile_values.extend(str(value) for value in values if str(value).strip())
	if isinstance(analysis, dict):
		analysis_data = analysis.get("data")
		if isinstance(analysis_data, dict):
			analysis_goal = analysis_data.get("career_goal")
			if analysis_goal and analysis_goal != "Not provided":
				profile_values.append(str(analysis_goal))
	profile_values.extend(sorted(user_skills))
	profile_document = " ".join(str(value) for value in profile_values if str(value).strip())

	role_names, vectorizer, role_vectors = get_career_tfidf_model()
	similarities = [0.0] * len(role_names)
	if profile_document.strip() and vectorizer.build_analyzer()(profile_document):
		profile_vector = vectorizer.transform([profile_document])
		similarities = cosine_similarity(profile_vector, role_vectors).ravel().tolist()

	career_results: list[dict[str, Any]] = []
	for name, payload, similarity in zip(role_names, CAREER_DATASET.values(), similarities):
		required = payload["required_skills"]
		matching, missing, partial, gap = career_match_score(user_skills, required)
		score = round(float(similarity) * 100)
		why = (
			"The ML text-similarity model found relevant overlap between your profile/resume and this role."
			if similarity > 0 else
			"Your profile/resume has little text overlap with this role so far; review the listed required skills."
		)
		recommendation = {
			"career_name": name,
			"recommendation_score": score,
			"why_suitable": why,
			"matching_skills": matching,
			"missing_skills": missing,
			"required_skills": required,
			"learning_direction": payload["learning_direction"],
			"overview": payload["overview"],
			"partial_skills": partial,
			"gap_percentage": gap,
		}
		career_results.append(recommendation)
	career_results.sort(key=lambda item: (-item["recommendation_score"], item["career_name"]))
	return career_results


def save_career_recommendations(user_id: int, recommendations: list[dict[str, Any]]) -> None:
	with connection() as database:
		database.execute("DELETE FROM user_career_recommendations WHERE user_id = ?", (user_id,))
		for item in recommendations:
			database.execute(
				"INSERT INTO user_career_recommendations (user_id, career_name, recommendation_score, why_suitable, matching_skills_json, missing_skills_json, required_skills_json, learning_direction, overview) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(user_id, item["career_name"], item["recommendation_score"], item["why_suitable"], json.dumps(item["matching_skills"]), json.dumps(item["missing_skills"]), json.dumps(item["required_skills"]), item["learning_direction"], item["overview"]),
			)


def get_recommendations_for_user(user_id: int) -> list[dict[str, Any]]:
	recommendations = compute_career_recommendations(user_id)
	with connection() as database:
		rows = database.execute(
			"SELECT * FROM user_career_recommendations WHERE user_id = ? ORDER BY recommendation_score DESC, career_name ASC",
			(user_id,),
		).fetchall()
	stored_items = [
		{
			"career_name": row["career_name"],
			"recommendation_score": row["recommendation_score"],
			"why_suitable": row["why_suitable"],
			"matching_skills": json.loads(row["matching_skills_json"]),
			"missing_skills": json.loads(row["missing_skills_json"]),
			"required_skills": json.loads(row["required_skills_json"]),
			"learning_direction": row["learning_direction"],
			"overview": row["overview"],
		}
		for row in rows
	]
	if len(stored_items) == len(recommendations) and all(
		stored_item == {key: recommendation[key] for key in stored_item}
		for stored_item, recommendation in zip(stored_items, recommendations)
	):
		return stored_items
	save_career_recommendations(user_id, recommendations)
	return recommendations


def skill_gap_analysis(user_id: int, career_name: str) -> dict[str, Any]:
	profile = get_user_profile(user_id)
	user_skills = get_user_skill_snapshot(user_id)
	payload = get_role_dataset(career_name)
	required = payload.get("required_skills", [])
	matching, missing, partial, gap = career_match_score(user_skills, required)
	analysis = get_analysis(user_id)
	skill_rows = analysis.get("skills_list", []) if isinstance(analysis, dict) else []
	priority_missing = [
		{"skill": skill, "priority": "High" if idx < 3 else "Medium", "category": "Required"}
		for idx, skill in enumerate(missing[:6])
	]
	return {
		"career_name": career_name,
		"matching_skills": matching,
		"missing_skills": missing,
		"partially_matched": partial,
		"skill_gap_percentage": gap,
		"priority_missing": priority_missing,
		"current_skill_level": "Profile and resume evidence available" if profile.get("skills") or any(row.get("source") in ("resume", "profile") for row in skill_rows) else "Not yet measured",
		"required_skill_level": "Target-ready skill set" if required else "Not defined",
		"career_description": payload.get("description", ""),
	}


def build_career_roadmap(user_id: int, career_name: str) -> list[dict[str, Any]]:
	user_skills = get_user_skill_snapshot(user_id)
	payload = get_role_dataset(career_name)
	stages: list[dict[str, Any]] = []
	for stage in payload.get("roadmap_stages", []):
		skills = stage.get("skills", [])
		matched = [skill for skill in skills if normalize_skill_name(skill) in user_skills]
		completion = round((len(matched) / max(1, len(skills))) * 100)
		stages.append({
			"stage_name": stage["stage"],
			"topics": stage.get("topics", []),
			"skills_to_learn": skills,
			"suggested_projects": stage.get("projects", []),
			"expected_outcome": stage.get("outcome", ""),
			"completion": completion,
		})
	return stages


def build_learning_path(user_id: int, career_name: str) -> list[dict[str, Any]]:
	user_skills = get_user_skill_snapshot(user_id)
	payload = get_role_dataset(career_name)
	path: list[dict[str, Any]] = []
	for item in payload.get("learning_topics", []):
		related = [skill for skill in payload.get("required_skills", []) if normalize_skill_name(skill) in normalize_skill_name(item) or normalize_skill_name(item) in normalize_skill_name(skill)]
		path.append({
			"topic": item,
			"reason": f"Supports the {career_name} roadmap and closes a core gap in your current learning path.",
			"priority": "High" if item in ["Machine Learning", "Deep Learning", "FastAPI", "NLP", "Docker"] else "Medium",
			"estimated_stage": "Foundation" if item in ["Python Advanced", "SQL", "HTML", "CSS"] else "Core",
			"related_skills": related or payload.get("required_skills", [])[:3],
			"status": "Completed" if normalize_skill_name(item) in user_skills else "Recommended",
		})
	return path


def get_career_required_skills(career_name: str) -> list[str]:
	payload = get_role_dataset(career_name)
	return payload.get("required_skills", [])


def get_future_scope(career_name: str) -> dict[str, Any]:
	payload = get_role_dataset(career_name)
	return {
		"career_growth_direction": payload.get("learning_direction", ""),
		"possible_advanced_roles": payload.get("future_roles", []),
		"skills_to_watch": payload.get("related_technologies", []),
		"advanced_learning_direction": payload.get("learning_direction", ""),
	}


def get_career_comparison(user_id: int, career_names: list[str]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for career_name in career_names:
		gap = skill_gap_analysis(user_id, career_name)
		recommendation = next((item for item in get_recommendations_for_user(user_id) if item["career_name"] == career_name), None)
		rows.append({
			"Career Name": career_name,
			"Required Skills": ", ".join(gap["matching_skills"] + gap["missing_skills"]),
			"User Skill Match": len(gap["matching_skills"]),
			"Skill Gap": gap["skill_gap_percentage"],
			"Learning Difficulty": "Moderate" if len(gap["missing_skills"]) <= 4 else "High",
			"Recommended Learning Areas": ", ".join(get_role_dataset(career_name).get("learning_topics", [])[:4]),
			"Future Scope": ", ".join(get_future_scope(career_name)["possible_advanced_roles"][:3]),
			"Compatibility": recommendation["recommendation_score"] if recommendation else 0,
		})
	return rows


def career_guidance_placeholder(user: dict[str, Any]) -> dict[str, Any]:
	user_id = int(user["id"])
	recommendations = get_recommendations_for_user(user_id)
	if not recommendations:
		recommendations = compute_career_recommendations(user_id)
		if recommendations:
			save_career_recommendations(user_id, recommendations)
	selected = recommendations[0]["career_name"] if recommendations else "Python Developer"
	return {
		"career_recommendations": recommendations,
		"selected_career": selected,
		"skill_gap": skill_gap_analysis(user_id, selected),
		"roadmap": build_career_roadmap(user_id, selected),
		"learning_path": build_learning_path(user_id, selected),
		"future_scope": get_future_scope(selected),
		"comparison": get_career_comparison(user_id, [item["career_name"] for item in recommendations[:4]]),
	}


# TAB: Career Guidance
# This page handles career matching, skill-gap analysis, roadmap, and recommendations.
def render_career_guidance_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	st.header("Career Guidance")
	st.caption("Career paths are ranked with scikit-learn TF-IDF text similarity against your profile and resume. Scores indicate relevance, not a probability or a model trained on user outcomes.")
	context = career_guidance_placeholder(user)
	recommendations = context["career_recommendations"]
	if not recommendations:
		st.info("We need a profile or resume analysis before we can recommend suitable IT careers.", icon=":material/psychology:")
		return
	selected_name = st.selectbox(
		"Selected career",
		options=[item["career_name"] for item in recommendations],
		index=0,
		key=f"career_guidance_select_{user_id}",
	)
	selected = next(item for item in recommendations if item["career_name"] == selected_name)
	selected_gap = skill_gap_analysis(user_id, selected_name)
	selected_roadmap = build_career_roadmap(user_id, selected_name)
	selected_path = build_learning_path(user_id, selected_name)
	selected_scope = get_future_scope(selected_name)

	intro_col, summary_col = st.columns([1.8, 1.2])
	with intro_col:
		with st.container(border=True):
			st.markdown(f"### {selected_name}")
			st.markdown(f"**Fit score:** {selected['recommendation_score']}%")
			st.write(selected["overview"])
			st.caption(f"Learning direction: {selected['learning_direction']}")
	with summary_col:
		with st.container(border=True):
			st.markdown("**Readiness snapshot**")
			st.metric("Skill match", f"{max(0, 100 - selected_gap['skill_gap_percentage'])}%", border=True)
			st.progress(max(0, 100 - selected_gap["skill_gap_percentage"]) / 100, text=f"Current readiness for {selected_name}")
			st.caption(f"Matching skills: {', '.join(selected['matching_skills']) if selected['matching_skills'] else 'No direct overlap detected'}")

	metric_cols = st.columns(4)
	with metric_cols[0]:
		st.metric("Recommendation", f"{selected['recommendation_score']}%", border=True)
	with metric_cols[1]:
		st.metric("Skill Gap", f"{selected_gap['skill_gap_percentage']}%", border=True)
	with metric_cols[2]:
		st.metric("Missing Skills", str(len(selected_gap['missing_skills'])), border=True)
	with metric_cols[3]:
		st.metric("Roadmap Stages", str(len(selected_roadmap)), border=True)

	st.subheader("Career matches")
	recommandation_cols = st.columns(2)
	for index, item in enumerate(recommendations[:4]):
		with recommandation_cols[index % 2]:
			with st.container(border=True):
				st.markdown(f"### {item['career_name']}")
				st.metric("Match Score", f"{item['recommendation_score']}%", border=True)
				st.write(item["why_suitable"])
				st.caption(f"Top gaps: {', '.join(item['missing_skills'][:4]) if item['missing_skills'] else 'No major gaps'}")

	st.subheader("Skill gap analysis")
	matching_col, missing_col, partial_col = st.columns(3)
	with matching_col:
		with st.container(border=True):
			st.markdown("**Matching skills**")
			st.write(", ".join(selected_gap["matching_skills"]) if selected_gap["matching_skills"] else "No strong matches found.")
	with missing_col:
		with st.container(border=True):
			st.markdown("**Missing skills**")
			st.write(", ".join(selected_gap["missing_skills"]) if selected_gap["missing_skills"] else "No critical gaps were identified.")
	with partial_col:
		with st.container(border=True):
			st.markdown("**Partially matched skills**")
			st.write(", ".join(selected_gap["partially_matched"]) if selected_gap["partially_matched"] else "No partial matches identified.")
	if selected_gap["priority_missing"]:
		st.dataframe(selected_gap["priority_missing"], hide_index=True)

	roadmap_col, learning_col = st.columns([1.25, 1.0])
	with roadmap_col:
		st.subheader("Career roadmap")
		for stage in selected_roadmap:
			with st.container(border=True):
				st.markdown(f"**{stage['stage_name']}**")
				st.caption(f"Progress: {stage['completion']}% · {stage['expected_outcome']}")
				st.write("Topics: " + ", ".join(stage["topics"]))
				st.write("Skills to learn: " + ", ".join(stage["skills_to_learn"]))
				st.write("Suggested projects: " + ", ".join(stage["suggested_projects"]))
				st.progress(stage["completion"] / 100)
	with learning_col:
		st.subheader("Learning path")
		for step in selected_path:
			with st.container(border=True):
				st.markdown(f"**{step['topic']}**")
				st.caption(f"Priority: {step['priority']} · Stage: {step['estimated_stage']} · Status: {step['status']}")
				st.write(step["reason"])

	future_col, required_col = st.columns(2)
	with future_col:
		st.subheader("Future scope")
		future = selected_scope
		with st.container(border=True):
			st.markdown("**Career growth direction**")
			st.write(future["career_growth_direction"])
			st.markdown("**Possible advanced roles**")
			st.write(", ".join(future["possible_advanced_roles"]))
			st.markdown("**Skills to watch**")
			st.write(", ".join(future["skills_to_watch"]))
	with required_col:
		st.subheader("Required skills")
		required_skills = get_career_required_skills(selected_name)
		user_skill_lookup = {normalize_skill_name(skill): skill for skill in get_user_skill_snapshot(user_id)}
		with st.container(border=True):
			for skill in required_skills:
				normal = normalize_skill_name(skill)
				if normal in user_skill_lookup:
					label = "✓ Already have"
				elif any(token in user_skill_lookup for token in [normal[:3], normal[:4], normal[:5]]):
					label = "⚠ Improve"
				else:
					label = "✕ Missing"
				st.write(f"{skill}: {label}")

	st.subheader("Career comparison")
	comparison_rows = get_career_comparison(user_id, [item["career_name"] for item in recommendations[:4]])
	if comparison_rows:
		st.dataframe(comparison_rows, hide_index=True)

	st.subheader("Salary insights – coming soon")
	st.info("This section is reserved for future integration with reliable external salary data sources and API-based earnings insights.", icon=":material/attach_money:")

	st.subheader("Future features")
	left, right = st.columns(2)
	with left:
		with st.container(border=True):
			st.markdown("**AI Resume Builder**")
			st.caption("Coming Soon")
	with right:
		with st.container(border=True):
			st.markdown("**AI Cover Letter Generator**")
			st.caption("Coming Soon")


# TAB: Learning Hub
# This page displays courses, topics, progress, and career-aligned learning tracks.
def render_learning_hub_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	ensure_learning_user_progress(user_id)
	st.header("Learning Hub")
	st.caption("Pick a course, complete guided topics, and build evidence-backed momentum toward your next career role.")
	categories = list_learning_categories()
	available_courses = list_learning_courses()
	if not categories or not available_courses:
		st.info("Learning content is being prepared. Please refresh or check the database setup.")
		return
	recommended_courses = get_learning_recommendations_for_user(user_id)
	recommended_names = [item["course_name"] for item in recommended_courses if item.get("course_name") in {course["course_name"] for course in available_courses}]
	category_names = ["All categories"] + [category["name"] for category in categories]
	selected_category = st.selectbox("Browse by category", category_names, index=0, key=f"learning_hub_category_{user_id}")
	if selected_category == "All categories":
		filtered_courses = available_courses
	else:
		filtered_courses = [course for course in available_courses if course["category_name"] == selected_category]
	if not filtered_courses:
		st.info("No courses are available in the selected category yet.")
		return
	course_names = [course["course_name"] for course in filtered_courses]
	preferred_course_name = st.session_state.get(f"learning_hub_selected_course_{user_id}")
	if preferred_course_name in course_names:
		selected_index = course_names.index(preferred_course_name)
	else:
		selected_index = 0
	selected_course_name = st.selectbox("Course", course_names, index=selected_index, key=f"learning_hub_course_{user_id}")
	st.session_state[f"learning_hub_selected_course_{user_id}"] = selected_course_name
	course = get_learning_course_details(selected_course_name)
	if not course:
		st.warning("This course is not available right now. Please choose another one.")
		return
	with connection() as database:
		course_progress = database.execute(
			"SELECT progress_percent, status FROM learning_user_course_progress WHERE user_id = ? AND course_id = (SELECT id FROM learning_courses WHERE course_name = ?)",
			(user_id, selected_course_name),
		).fetchone()
		progress_value = int(course_progress["progress_percent"]) if course_progress else 0
		progress_status = course_progress["status"] if course_progress else "Not started"
	st.subheader(f"{course['course_name']}" )
	meta_col, progress_col = st.columns([1.7, 1.3])
	with meta_col:
		with st.container(border=True):
			st.markdown(f"**{course['category_name']}**")
			st.caption(f"{course['difficulty']} · {course['duration']}")
			st.write(course["description"])
	with progress_col:
		with st.container(border=True):
			st.markdown("**Progress**")
			st.metric("Completion", f"{progress_value}%", border=True)
			st.progress(progress_value / 100, text=progress_status)
			st.caption(f"Current status: {progress_status}")

	recommended_cards = [
		{
			"title": "Python",
			"description": "Learn Python basics, logic building, and scripting workflows.",
			"url": "https://www.w3schools.com/python/",
		},
		{
			"title": "SQL",
			"description": "Practice databases, queries, and data handling with SQL.",
			"url": "https://www.w3schools.com/sql/",
		},
		{
			"title": "HTML",
			"description": "Build web page structure with semantic markup and content layout.",
			"url": "https://www.w3schools.com/html/",
		},
		{
			"title": "CSS",
			"description": "Style responsive interfaces and modern page layouts.",
			"url": "https://www.w3schools.com/css/",
		},
		{
			"title": "JavaScript",
			"description": "Add interactivity, events, and dynamic behavior to websites.",
			"url": "https://www.w3schools.com/js/",
		},
		{
			"title": "Cyber Security",
			"description": "Learn security fundamentals, threats, and digital protection.",
			"url": "https://www.w3schools.com/cybersecurity/",
		},
		{
			"title": "AI Fundamentals",
			"description": "Understand AI concepts, workflows, and practical use cases.",
			"url": "https://www.w3schools.com/ai/",
		},
		{
			"title": "Web Development",
			"description": "Explore full web development fundamentals across core frontend topics.",
			"url": "https://www.w3schools.com/",
		},
	]
	st.subheader("Recommended for you")
	rec_cols = st.columns(min(4, len(recommended_cards)))
	for index, rec in enumerate(recommended_cards):
		with rec_cols[index % min(4, len(recommended_cards))]:
			with st.container(border=True):
				st.markdown(f"**{rec['title']}**")
				st.caption(rec["description"])
				st.link_button("Open", rec["url"], use_container_width=True)

	st.subheader("Topics")
	topic_columns = st.columns(2)
	for index, topic in enumerate(course["topics"]):
		with topic_columns[index % 2]:
			with st.container(border=True):
				st.markdown(f"### {topic['title']}")
				st.write(topic["explanation"])
				st.markdown("**Key notes**")
				for note in topic["notes"][:3]:
					st.write(f"• {note}")
				st.markdown("**Example**")
				st.code(topic["examples"][0] if topic["examples"] else "# Example")
				if st.button("Mark topic complete", key=f"learning_topic_complete_{user_id}_{topic['title']}"):
					update_topic_progress(user_id, selected_course_name, topic["title"], 100, "Completed")
					st.success(f"{topic['title']} marked as complete.")
					st.rerun()
				if topic.get("lessons"):
					with st.expander("View lessons"):
						for lesson in topic["lessons"]:
							st.markdown(f"**{lesson['lesson_title']}**")
							st.write(lesson["lesson_content"])

	st.subheader("Projects")
	for project in course["projects"]:
		with st.container(border=True):
			st.markdown(f"### {project['title']}")
			st.caption(f"{project['difficulty']} · {' · '.join(project['technologies'])}")
			st.write(project["objective"])
			st.markdown("**What you will build**")
			for step in project["steps"]:
				st.write(f"• {step}")
			st.markdown("**Outcomes**")
			st.write(project["expected_outcome"])

	st.subheader("Learning checklist")
	with st.container(border=True):
		for topic in course["topics"]:
			st.checkbox(f"{topic['title']}", value=False, key=f"learning_check_{user_id}_{topic['title']}")

	st.subheader("Career-aligned next steps")
	with st.container(border=True):
		recommended_set = [course_name for course_name in recommended_names[:3]]
		if recommended_set:
			st.write("Prioritize these tracks next: " + ", ".join(recommended_set))
		else:
			st.write("Continue with your current track and keep building core skills in Python, SQL, and product thinking.")


def generate_resume_report(user: dict[str, Any], analysis: dict[str, Any]) -> bytes:
	from io import BytesIO

	buffer = BytesIO()
	document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
	styles = getSampleStyleSheet()
	styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], textColor=colors.HexColor("#087E78"), alignment=TA_CENTER, fontSize=20, spaceAfter=4))
	styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#183B3A"), spaceBefore=12, spaceAfter=5))
	styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=13))
	story = [Paragraph("CareerAI", styles["Brand"]), Paragraph("Resume analysis report", styles["Heading2"]), Spacer(1, 8)]
	data = analysis["data"]
	story.append(Paragraph("Candidate and resume", styles["SectionTitle"]))
	for label, value in (("Account", user.get("full_name", "")), ("Email", user.get("email", "")), ("Resume", analysis["original_filename"]), ("Analysis date", analysis["created_at"]), ("Career goal", data.get("career_goal", "Not provided"))):
		story.append(Paragraph(f"<b>{label}:</b> {reportlab_escape(str(value))}", styles["BodySmall"]))
	story.append(Paragraph(f"Overall ATS score: <b>{analysis['overall']}/100</b>", styles["SectionTitle"]))
	score_keys = [("Skills", "skills"), ("Keywords", "keywords"), ("Structure", "structure"), ("Education", "education"), ("Experience", "experience"), ("Projects", "projects"), ("Completeness", "completeness")]
	score_table = [[label, f"{analysis[key]}/100"] for label, key in score_keys]
	table = Table(score_table, colWidths=[3.1 * inch, 1.1 * inch])
	table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F6F5")), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#183B3A")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E2DF")), ("PADDING", (0, 0), (-1, -1), 6)]))
	story.extend([table, Paragraph("Detected skills", styles["SectionTitle"])])
	story.append(Paragraph(reportlab_escape(", ".join(row["skill"] for row in analysis["skills_list"] if row["source"] == "resume") or "No catalogued skills detected"), styles["BodySmall"]))
	for title, key in (("Missing skills for career goal", "missing"), ("Recommended skills", "recommended")):
		items = [row["skill"] for row in analysis["skills_list"] if row["source"] == key]
		story.extend([Paragraph(title, styles["SectionTitle"]), Paragraph(reportlab_escape(", ".join(items) or "None inferred"), styles["BodySmall"])])
	for title, values in (("Resume strengths", data.get("strengths", [])), ("Resume weak areas", data.get("weak_areas", [])), ("Personalized suggestions", [f"{item['priority']}: {item['text']}" for item in analysis.get("suggestions", [])])):
		story.append(Paragraph(title, styles["SectionTitle"]))
		for value in values or ["None identified."]:
			story.append(Paragraph("• " + reportlab_escape(str(value)), styles["BodySmall"]))
	document.build(story)
	return buffer.getvalue()


def reportlab_escape(value: str) -> str:
	return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_generated_report(user_id: int, analysis_id: int, content: bytes) -> int:
	REPORT_STORAGE.mkdir(parents=True, exist_ok=True)
	storage_key = write_private_file(REPORT_STORAGE, user_id, content)
	try:
		with connection() as database:
			cursor = database.execute(
				"INSERT INTO generated_reports (user_id, analysis_id, storage_key) VALUES (?, ?, ?)",
				(user_id, analysis_id, storage_key),
			)
			return int(cursor.lastrowid)
	except Exception:
		private_file_path(REPORT_STORAGE, storage_key, user_id).unlink(missing_ok=True)
		raise


def get_generated_report(user_id: int, report_id: int) -> bytes:
	with connection() as database:
		row = database.execute("SELECT storage_key FROM generated_reports WHERE id = ? AND user_id = ?", (report_id, user_id)).fetchone()
	if not row:
		raise PermissionError("Report not found for this account.")
	return private_file_path(REPORT_STORAGE, row["storage_key"], user_id).read_bytes()


@st.cache_resource
def get_resume_nlp():
	nlp = spacy.blank("en")
	nlp.add_pipe("sentencizer")
	return nlp


def perform_resume_analysis(user: dict[str, Any], resume_id: int) -> int:
	content = get_resume_bytes(user["id"], resume_id)
	text = extract_pdf_text(content)
	if not text.strip():
		raise ValueError("No readable text was found. Please upload a text-based PDF rather than a scanned image.")
	with st.spinner("Reading sections, skills, and ATS signals…"):
		parsed = get_resume_nlp()(text)
		cleaned_text = "\n".join(sentence.text for sentence in parsed.sents)
		result = analyze_resume_text(cleaned_text, user["full_name"], get_profile_context(user["id"]))
	return persist_resume_analysis(user["id"], resume_id, result)


def render_resume_skills(rows: list[dict[str, Any]], source: str) -> None:
	selected = [row for row in rows if row["source"] == source]
	if not selected:
		st.caption("None identified from the available evidence.")
		return
	grouped: dict[str, list[str]] = {}
	for row in selected:
		grouped.setdefault(row["category"], []).append(row["skill"])
	for category, skills in grouped.items():
		st.markdown(f"**{category}**")
		st.write(" · ".join(skills))


# TAB: Resume
# This page handles resume upload, ATS analysis, skill extraction, and PDF report generation.
def render_resume_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	st.header("Resume")
	st.caption("Analyze your resume, understand ATS signals, and keep a private history of each version.")

	current = get_current_resume(user_id)
	upload_column, current_column = st.columns([1.5, 1], vertical_alignment="top")
	with upload_column:
		with st.container(border=True):
			st.subheader("Upload resume")
			st.caption("PDF only · Maximum 10 MB · Files are stored privately.")
			uploaded = st.file_uploader(
				"Choose a PDF resume",
				type=["pdf"],
				key=f"resume_pdf_upload_{user_id}",
				help="Text-based PDFs provide the most complete analysis.",
			)
			if uploaded:
				if st.button(
					"Upload and analyze",
					type="primary",
					icon=":material/analytics:",
					key="upload_analyze_resume",
					width="stretch",
				):
					try:
						resume_id = upload_resume(user_id, uploaded.name, uploaded.getvalue())
						analysis_id = perform_resume_analysis(user, resume_id)
						st.session_state["resume_selected_analysis"] = analysis_id
						st.session_state.pop("resume_report_ref", None)
						st.success("Resume uploaded and analyzed. Previous analysis history is preserved.")
						st.rerun()
					except (ValueError, PermissionError, OSError, fitz.FileDataError) as error:
						st.error(str(error))
			elif not current:
				st.info("Upload your first PDF resume to see your analysis here.", icon=":material/upload_file:")
	with current_column:
		with st.container(border=True):
			st.subheader("Current resume")
			if current:
				st.caption(
					f"**{current['original_filename']}**\n\n"
					f"{current['file_size'] / 1024:.0f} KB · Uploaded {current['uploaded_at']}"
				)
				if not uploaded and st.button(
					"Analyze current resume",
					type="primary",
					icon=":material/refresh:",
					key="analyze_current_resume",
					width="stretch",
				):
					try:
						analysis_id = perform_resume_analysis(user, current["id"])
						st.session_state["resume_selected_analysis"] = analysis_id
						st.session_state.pop("resume_report_ref", None)
						st.success("New analysis saved to your history.")
						st.rerun()
					except (ValueError, PermissionError, OSError, fitz.FileDataError) as error:
						st.error(str(error))
			else:
				st.caption("No resume uploaded yet.")

	history = list_resume_history(user_id)
	if not history:
		with st.container(border=True):
			st.subheader("Your analysis will appear here")
			st.caption("Your extracted overview, ATS score, skills, and personalized recommendations will appear after your first analysis.")
		return

	history_labels = {item["id"]: f"{item['created_at']} · {item['original_filename']} · ATS {item['overall']}" for item in history}
	valid_ids = list(history_labels)
	selected_id = st.session_state.get("resume_selected_analysis")
	if selected_id not in history_labels:
		selected_id = valid_ids[0]
		st.session_state["resume_selected_analysis"] = selected_id
	with st.container(border=True):
		choice = st.selectbox(
			"Viewing analysis",
			options=valid_ids,
			format_func=lambda item_id: history_labels[item_id],
			key=f"resume_history_picker_{user_id}",
		)
	analysis = get_analysis(user_id, int(choice))
	if not analysis:
		st.error("This analysis is not available for the current account.")
		return

	overview_tab, match_tab, skills_tab, reports_tab = st.tabs(
		["Overview", "Job match", "Skills & suggestions", "Reports & history"],
	)

	with overview_tab:
		st.subheader("Resume overview")
		metric_columns = st.columns(4)
		for column, label, value in zip(
			metric_columns,
			("Candidate", "Email", "Phone", "Resume length"),
			(
				analysis["data"]["name"],
				analysis["data"]["email"],
				analysis["data"]["phone"],
				f"{analysis['data']['word_count']} words",
			),
		):
			with column:
				st.metric(label, value if len(value) <= 28 else value[:25] + "…", border=True)

		score_column, sections_column = st.columns([1, 1.2], vertical_alignment="top")
		with score_column:
			with st.container(border=True):
				st.subheader("ATS score")
				score_labels = ("Overall ATS", "Skills", "Keywords", "Structure")
				score_keys = ("overall", "skills", "keywords", "structure")
				for start in range(0, len(score_labels), 2):
					score_columns = st.columns(2)
					for column, label, score_key in zip(
						score_columns,
						score_labels[start:start + 2],
						score_keys[start:start + 2],
					):
						with column:
							st.metric(label, f"{analysis[score_key]}/100", border=True)
				st.progress(analysis["overall"] / 100, text=f"Overall score: {analysis['overall']} / 100")
				score_labels = {
					"skills": "Skills",
					"keywords": "Keywords",
					"structure": "Structure",
					"education": "Education",
					"experience": "Experience",
					"projects": "Projects",
					"completeness": "Completeness",
				}
				score_chart = [{"Dimension": label, "Score": analysis[key]} for key, label in score_labels.items()]
				st.bar_chart(score_chart, x="Dimension", y="Score", horizontal=True, height=250)
		with sections_column:
			with st.container(border=True):
				st.subheader("Detected skills")
				render_resume_skills(analysis["skills_list"], "resume")
				if any(row["source"] == "profile" for row in analysis["skills_list"]):
					st.markdown("**Supporting skills from My Profile**")
					render_resume_skills(analysis["skills_list"], "profile")
			with st.expander("Sections and extracted details", icon=":material/subject:"):
				available_column, missing_column = st.columns(2)
				with available_column:
					st.markdown("**Detected sections**")
					st.write(" · ".join(analysis["sections"]["available"]) or "No standard section headings detected")
				with missing_column:
					st.markdown("**Sections not detected**")
					st.write(" · ".join(analysis["sections"]["missing"]) or "No common sections appear to be missing")
				st.markdown("**Education**")
				st.write("\n".join(analysis["data"]["education"]) or "No education details detected")
				st.markdown("**Experience and projects**")
				st.write("\n".join(analysis["data"]["experience"] + analysis["data"]["projects"]) or "No experience or project details detected")
				st.markdown("**Certifications and achievements**")
				st.write("\n".join(analysis["data"]["certifications"] + analysis["data"]["achievements"]) or "No certifications or measurable achievements detected")
				st.markdown("**Relevant keywords**")
				st.write(" · ".join(analysis["data"]["keywords"]) or "No catalogued keywords detected")
				st.markdown("**Soft skills**")
				st.write(" · ".join(analysis["data"].get("soft_skills", [])) or "No catalogued soft skills detected")
				st.markdown("**Strengths**")
				for value in analysis["data"]["strengths"]:
					st.markdown(f"- {value}")
				st.markdown("**Weak areas**")
				for value in analysis["data"]["weak_areas"] or ["No high-priority weak areas were identified."]:
					st.markdown(f"- {value}")
		st.caption("Scores use transparent text, section, and skill-catalogue signals; they are guidance, not a hiring decision or a trained prediction model.")

	with match_tab:
		st.subheader("Match against a job description")
		st.caption("Compare skills and experience evidence from this resume with a role description. The match score reflects recognized skill overlap only.")
		match_result_key = f"resume_jd_match_result_{user_id}_{analysis['id']}"
		with st.form(f"resume_jd_match_form_{user_id}_{analysis['id']}"):
			job_description = st.text_area(
				"Paste the job description",
				placeholder="Paste the responsibilities and qualifications for the role...",
				max_chars=20000,
				height=180,
				key=f"resume_jd_match_text_{user_id}_{analysis['id']}",
			)
			match_submitted = st.form_submit_button(
				"Compare resume with job",
				type="primary",
				icon=":material/compare_arrows:",
			)
		if match_submitted:
			st.session_state.pop(match_result_key, None)
			if len(job_description.strip()) < 30:
				st.error("Paste a job description with at least 30 characters to run the comparison.")
			else:
				try:
					resume_text = extract_pdf_text(get_resume_bytes(user_id, int(analysis["resume_id"])))
					if not resume_text:
						raise ValueError("No readable text was found in this resume PDF.")
					st.session_state[match_result_key] = compare_resume_to_job_description(
						resume_text,
						job_description.strip(),
						analysis["data"]["name"],
					)
				except (ValueError, PermissionError, OSError, fitz.FileDataError) as error:
					st.error(f"Could not compare this resume: {error}")

		match_result = st.session_state.get(match_result_key)
		if match_result:
			if match_result["match_score"] is None:
				st.info("No skills from CareerAI's skill catalogue were found in this job description, so a skill-overlap score is not available.")
			else:
				st.metric(
					"Recognized skill overlap",
					f"{match_result['match_score']}%",
					help=f"{len(match_result['matched_skills'])} of {len(match_result['required_skills'])} catalogue skills found in both texts.",
				)
			match_columns = st.columns(2)
			with match_columns[0]:
				with st.container(border=True):
					st.markdown(f"**Matched skills ({len(match_result['matched_skills'])})**")
					st.write(" · ".join(match_result["matched_skills"]) or "No catalogue-recognized skill overlap found.")
			with match_columns[1]:
				with st.container(border=True):
					st.markdown(f"**Skills not detected ({len(match_result['missing_skills'])})**")
					st.write(" · ".join(match_result["missing_skills"]) or "All catalogue-recognized job skills appear in the resume.")

			st.subheader("Experience and project evidence")
			experience_and_projects = [
				("Experience", line) for line in match_result["experience"]
			] + [
				("Project", line) for line in match_result["projects"]
			]
			if experience_and_projects:
				for source, line in experience_and_projects:
					st.markdown(f"- **{source}:** {line}")
			else:
				st.caption("No experience or project lines were detected by the current text-based section parser.")

			st.subheader("Job-description keyword overlap")
			if match_result["evidence"]:
				for item in match_result["evidence"]:
					st.markdown(f"- **{item['source']}:** {item['line']}")
					st.caption("Matched terms: " + ", ".join(item["matched_terms"]))
			else:
				st.caption("No exact keyword overlap was found in the extracted experience or project lines.")

			st.subheader("Suggestions for this role")
			for suggestion in match_result["suggestions"]:
				st.markdown(f"- {suggestion}")
			st.caption("This private, rule-based comparison checks a built-in skills catalogue and exact text evidence. The pasted job description is kept only in this session and is not saved to resume history.")
		else:
			st.info("Paste a job description above to see recognized skill overlap and evidence from your resume.", icon=":material/compare_arrows:")

	with skills_tab:
		st.subheader("Skills for your career goal")
		st.caption(f"Career goal: {analysis['data'].get('career_goal', 'Not provided')}")
		missing_column, recommended_column = st.columns(2)
		with missing_column:
			with st.container(border=True):
				st.markdown("**Missing for this career goal**")
				render_resume_skills(analysis["skills_list"], "missing")
		with recommended_column:
			with st.container(border=True):
				st.markdown("**Role skill reference**")
				render_resume_skills(analysis["skills_list"], "recommended")
		st.caption("Career gaps are shown only when a supported role map matches the profile goal.")

		st.subheader("Resume suggestions")
		for start in range(0, len(analysis["suggestions"]), 2):
			suggestion_columns = st.columns(2)
			for column, (index, item) in zip(
				suggestion_columns,
				enumerate(analysis["suggestions"][start:start + 2], start=start + 1),
			):
				with column:
					with st.container(border=True):
						st.badge(item["priority"])
						st.markdown(f"**Suggestion {index}**")
						st.write(item["text"])

	with reports_tab:
		st.subheader("Analysis history")
		st.dataframe(
			history,
			width="stretch",
			hide_index=True,
			column_config={
				"id": st.column_config.NumberColumn("Analysis"),
				"created_at": "Date",
				"original_filename": "Resume",
				"overall": st.column_config.ProgressColumn("ATS score", min_value=0, max_value=100, format="%d"),
			},
		)
		st.subheader("PDF report")
		if st.button("Generate PDF report", icon=":material/picture_as_pdf:", key=f"generate_report_{analysis['id']}"):
			try:
				report_bytes = generate_resume_report(user, analysis)
				report_id = save_generated_report(user_id, analysis["id"], report_bytes)
				st.session_state["resume_report_ref"] = (user_id, report_id, analysis["id"])
				st.success("Your report is ready to download.")
			except (OSError, PermissionError, ValueError) as error:
				st.error(f"Could not generate the report: {error}")
		report_ref = st.session_state.get("resume_report_ref")
		if report_ref and report_ref[0] == user_id and report_ref[2] == analysis["id"]:
			try:
				report_bytes = get_generated_report(user_id, report_ref[1])
				st.download_button(
					"Download analysis report",
					data=report_bytes,
					file_name=f"CareerAI_resume_report_{analysis['id']}.pdf",
					mime="application/pdf",
					icon=":material/download:",
					key=f"download_report_{report_ref[1]}",
				)
			except (OSError, PermissionError) as error:
				st.error(str(error))
		with st.expander("Coming soon"):
			future_columns = st.columns(2)
			with future_columns[0]:
				st.markdown("**AI Resume Builder**")
			with future_columns[1]:
				st.markdown("**AI Cover Letter Generator**")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def initialize_session() -> None:
	st.session_state.setdefault("auth_user", None)
	st.session_state.setdefault("auth_view", "login")


def inject_styles() -> None:
	st.markdown(
		"""
		<style>
		.stApp { background: #f5f7f8; color: #1f2933; }
		[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e1e7ea; }
		[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #44515c; }
		[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #687781; }
		[data-testid="stSidebar"] [data-testid="stButton"] button { min-height: 2.65rem; justify-content: flex-start; border-radius: 6px; }
		[data-testid="stSidebar"] [data-testid="stButton"] button p { text-align: left; }
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] { background: #f6f8f9 !important; color: #26343b !important; border: 1px solid #d8e0e4 !important; }
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] p,
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] span { color: #26343b !important; }
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] { background: #087e78 !important; color: #ffffff !important; border: 1px solid #087e78 !important; }
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] p,
		[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] span { color: #ffffff !important; }
		.brand-mark { color: #087e78; font-size: .9rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
		.auth-subtitle { color: #52616b; margin-bottom: 1.3rem; }
		[data-testid="stForm"] { border: 1px solid #d8e1e5; border-radius: 8px; background: #ffffff; padding: 1.25rem; }
		[data-testid="stMetric"] { border-radius: 8px; }
		[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
		</style>
		""",
		unsafe_allow_html=True,
	)


# TAB: Login
# Authentication screen for existing users to sign in.
def login_screen() -> None:
	with st.form("login_form"):
		st.subheader("Welcome back")
		st.caption("Sign in to continue your career journey.")
		email = st.text_input("Email address", placeholder="you@company.com", autocomplete="email")
		password = st.text_input("Password", type="password", placeholder="Enter your password", autocomplete="current-password")
		submitted = st.form_submit_button("Log in", type="primary", width="stretch", icon=":material/login:")
	if submitted:
		success, message, user = login_user(email, password)
		if success:
			st.session_state.auth_user = {key: user[key] for key in ("id", "full_name", "email", "role")}
			st.success(message)
			st.rerun()
		st.error(message, icon=":material/error:")


# TAB: Create Account
# Authentication screen for new user registration.
def signup_screen() -> None:
	with st.form("signup_form"):
		st.subheader("Create your account")
		st.caption("Build a stronger path from your experience to your next opportunity.")
		full_name = st.text_input("Full name", placeholder="Alex Morgan", autocomplete="name")
		email = st.text_input("Email address", placeholder="you@company.com", autocomplete="email")
		password = st.text_input("Password", type="password", placeholder="8+ characters with mixed complexity", autocomplete="new-password")
		confirmation = st.text_input("Confirm password", type="password", autocomplete="new-password")
		submitted = st.form_submit_button("Create account", type="primary", width="stretch", icon=":material/person_add:")
	if submitted:
		success, message, verification_token = register_user(full_name, email, password, confirmation)
		if success:
			st.success(message, icon=":material/check_circle:")
			if EXPOSE_DEV_TOKENS:
				st.info(f"Development verification token: `{verification_token}`")
			st.session_state.auth_view = "login"
		else:
			st.error(message, icon=":material/error:")


# TAB: Forgot Password
# Authentication screen for password reset flow.
def forgot_screen() -> None:
	with st.form("forgot_form"):
		st.subheader("Reset your password")
		st.caption("Enter your email and we will prepare a secure reset link.")
		email = st.text_input("Email address", placeholder="you@company.com", autocomplete="email")
		submitted = st.form_submit_button("Send reset instructions", type="primary", width="stretch", icon=":material/forward_to_inbox:")
	if submitted:
		message, reset_token = request_reset(email)
		st.success(message, icon=":material/mark_email_read:")
		if reset_token and EXPOSE_DEV_TOKENS:
			st.info(f"Development reset token: `{reset_token}`")
	with st.expander("Already have a reset token?", icon=":material/key:"):
		with st.form("reset_form"):
			raw_token = st.text_input("Reset token", type="password")
			password = st.text_input("New password", type="password", autocomplete="new-password")
			confirmation = st.text_input("Confirm new password", type="password", autocomplete="new-password")
			submitted = st.form_submit_button("Update password", width="stretch")
		if submitted:
			success, message = update_password(raw_token, password, confirmation)
			(st.success if success else st.error)(message)


# TAB: Email Verification
# Handles magic-link or token verification flow after signup or reset.
def verification_screen() -> None:
	with st.expander("Email verification", icon=":material/mark_email_unread:"):
		with st.form("verification_form"):
			raw_token = st.text_input("Verification token", type="password")
			submitted = st.form_submit_button("Verify email", type="primary", width="stretch", icon=":material/verified:")
		if submitted:
			success, message = verify_email(raw_token)
			(st.success if success else st.error)(message)


# TAB: Dashboard
# This page shows overview metrics, progress summary, and recent user activities.
def render_dashboard_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	profile = get_user_profile(user_id)
	summary = get_dashboard_resume_summary(user_id)
	history = list_resume_history(user_id)
	latest_score = summary["latest_ats_score"]
	latest_analysis = get_analysis(user_id)
	completion = profile_completion(user, profile)
	jobs_progress = get_jobs_report_progress(user_id)
	jobs_data = get_jobs_profile_data(user_id)
	dashboard_matches = [score_job_match(listing, jobs_data) for listing in JOB_SAMPLE_LISTINGS]
	top_job = next(
		(item for item in sorted(dashboard_matches, key=lambda item: item["match_score"], reverse=True) if item["type"] == "Job"),
		None,
	)
	top_internship = next(
		(item for item in sorted(dashboard_matches, key=lambda item: item["match_score"], reverse=True) if item["type"] == "Internship"),
		None,
	)
	recommendations = jobs_data["recommendations"]
	recent_activity = get_activity_history(user_id)[:5]

	st.title(f"Welcome back, {user['full_name'].split()[0]}")
	st.caption("Your career progress, resume insights, and next steps at a glance.")
	with st.container(horizontal=True):
		if st.button("Update profile", icon=":material/person:", key=f"dashboard_profile_{user_id}"):
			st.session_state.main_navigation = "My Profile"
			st.rerun()
		if st.button("Analyze resume", icon=":material/description:", type="primary", key=f"dashboard_resume_{user_id}"):
			st.session_state.main_navigation = "Resume"
			st.rerun()
		if st.button("Explore jobs & reports", icon=":material/workspace_premium:", key=f"dashboard_jobs_reports_{user_id}"):
			st.session_state.main_navigation = "Jobs & Reports"
			st.rerun()

	st.subheader("Your progress")
	with st.container(horizontal=True):
		st.metric(
			"Profile completion",
			f"{completion}%",
			border=True,
			chart_data=[completion],
			chart_type="bar",
		)
		st.metric(
			"Latest ATS score",
			f"{latest_score['overall']}/100" if latest_score else "Not analyzed",
			delta=(
				f"{latest_score['overall'] - history[1]['overall']:+d} pts"
				if latest_score and len(history) > 1 else None
			),
			border=True,
			chart_data=[row["overall"] for row in reversed(history[:8])],
			chart_type="line",
		)
		st.metric(
			"Career readiness",
			f"{jobs_progress['career_readiness']}%" if jobs_progress["career_readiness"] is not None else "Not enough data",
			border=True,
		)
		st.metric(
			"Learning progress",
			f"{jobs_progress['learning_progress']}%",
			delta=f"{jobs_progress['course_completed']} courses completed",
			border=True,
		)

	if completion < 100:
		st.progress(completion / 100, text=f"Complete your profile to improve personalized recommendations · {completion}%")

	st.space("small")
	st.subheader("Resume insights")
	left, right = st.columns(2)
	with left:
		with st.container(border=True):
			st.markdown("**ATS score history**")
			if history:
				period_label = st.selectbox(
					"Show analyses",
					["Last 5", "Last 10", "All available"],
					key=f"dashboard_period_{user_id}",
				)
				limit = {"Last 5": 5, "Last 10": 10, "All available": len(history)}[period_label]
				visible_history = list(reversed(history[:limit]))
				trend = [{"Analysis": f"#{row['id']}", "ATS score": row["overall"]} for row in visible_history]
				st.line_chart(trend, x="Analysis", y="ATS score", height=250)
				st.caption(f"{len(visible_history)} of {len(history)} saved analyses, oldest to newest.")
			else:
				st.info("Analyze a resume to start tracking your ATS score.")
	with right:
		with st.container(border=True):
			st.markdown("**Latest score breakdown**")
			if latest_analysis:
				dimensions = [
					("Skills", "skills"), ("Keywords", "keywords"), ("Structure", "structure"),
					("Education", "education"), ("Experience", "experience"),
					("Projects", "projects"), ("Completeness", "completeness"),
				]
				breakdown = [{"Dimension": label, "Score": latest_analysis[key]} for label, key in dimensions]
				st.bar_chart(breakdown, x="Dimension", y="Score", horizontal=True, height=250)
				st.caption("Based on your latest saved resume analysis.")
			else:
				st.info("Your score breakdown will appear after your first resume analysis.")

	st.subheader("Career direction")
	career_column, skills_column = st.columns([1, 1])
	with career_column:
		with st.container(border=True):
			st.markdown("**Your target role**")
			goal = profile["career_goal"] or (
				latest_analysis["data"].get("career_goal", "") if latest_analysis else ""
			)
			st.markdown(f"### {goal if goal and goal != 'Not provided' else 'Not set yet'}")
			if recommendations:
				st.caption("Based on your saved career recommendations")
				for item in recommendations[:3]:
					st.write(f"**{item['career_name']}** · {item['recommendation_score']}% match")
			elif goal:
				st.caption("Add profile skills or run Career Guidance to see saved role recommendations.")
			else:
				st.caption("Set a career goal in My Profile to get focused recommendations.")
	with skills_column:
		with st.container(border=True):
			st.markdown("**Skills identified in your latest resume**")
			if latest_analysis:
				category_counts: dict[str, int] = {}
				for item in latest_analysis["skills_list"]:
					if item["source"] == "resume":
						category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
				if category_counts:
					category_data = [
						{"Category": category, "Skills": count}
						for category, count in sorted(category_counts.items(), key=lambda entry: (-entry[1], entry[0]))
					]
					st.bar_chart(category_data, x="Category", y="Skills", horizontal=True, height=220)
				else:
					st.info("No catalogued skills were detected in the latest resume.")
			else:
				st.info("Analyze a resume to see your technical skill mix.")

	st.subheader("Opportunities and activity")
	opportunity_column, activity_column = st.columns([1, 1])
	with opportunity_column:
		with st.container(border=True):
			st.markdown("**Role matches**")
			if top_job:
				st.write(f"**Job example:** {top_job['role']} · {top_job['match_score']}% match")
			else:
				st.caption("No job example match available yet.")
			if top_internship:
				st.write(f"**Internship example:** {top_internship['role']} · {top_internship['match_score']}% match")
			else:
				st.caption("No internship example match available yet.")
			st.caption("Curated examples only—not live vacancies.")
			st.caption(
				"Resume report: ready" if latest_analysis else "Analyze a resume to enable your resume report."
			)
	with activity_column:
		with st.container(border=True):
			st.markdown("**Recent activity**")
			if recent_activity:
				for item in recent_activity:
					st.write(f"**{item['activity']}** · {item['module']}")
					st.caption(f"{item['created_at']} · {item['status']}")
			else:
				st.info("Your learning, resume, and practice activity will appear here.")


# TAB: My Profile
# This page manages user profile details, career goal, skills, and profile updates.
def render_profile_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	profile = get_user_profile(user_id)
	st.header("My Profile")
	st.caption("Keep your career goal and skills current so Resume analysis can make relevant recommendations.")
	with st.container(border=True):
		st.markdown(f"**{user['full_name']}**")
		st.caption(user["email"])
	with st.form(f"profile_form_{user_id}"):
		career_goal = st.text_input("Career goal", value=profile["career_goal"], placeholder="For example: Data Analyst")
		education = st.text_area("Education", value=profile["education"], placeholder="Degree, institution, and graduation year", height=90)
		experience = st.text_area("Experience", value=profile["experience_summary"], placeholder="Role, organization, and relevant experience", height=110)
		skills_text = st.text_input("Skills", value=", ".join(profile["skills"]), placeholder="Python, SQL, Excel")
		interests_text = st.text_input("Career interests", value=", ".join(profile["interests"]), placeholder="Analytics, machine learning")
		submitted = st.form_submit_button("Save profile", type="primary", icon=":material/save:")
	if submitted:
		updated_profile = {
			"career_goal": career_goal.strip(),
			"education": education.strip(),
			"experience_summary": experience.strip(),
			"skills": list(dict.fromkeys(item.strip() for item in skills_text.split(",") if item.strip())),
			"interests": list(dict.fromkeys(item.strip() for item in interests_text.split(",") if item.strip())),
		}
		save_user_profile(user_id, updated_profile)
		st.success("Profile saved. Resume analysis will use the updated career goal and skills.")
		st.rerun()
	st.progress(profile_completion(user, profile) / 100, text=f"Profile completion: {profile_completion(user, profile)}%")


def seed_resources_data() -> None:
	resource_categories = {
		"Python": "Core programming concepts and daily practice for problem solving and automation.",
		"SQL & Databases": "Querying, modelling, and understanding relational data systems.",
		"Web Development": "Frontend and backend content for complete web applications and interfaces.",
		"Data Analytics": "Reporting, dashboards, and evidence-backed decision-making workflows.",
		"AI & ML": "Practical artificial intelligence and machine learning foundations.",
		"Cybersecurity": "Threat awareness, secure workflows, and digital defence fundamentals.",
		"Cloud & DevOps": "Deployment pipelines, automation, and modern infrastructure operations.",
		"Interview Preparation": "Behavioral and technical interview prep with examples and quick practice.",
		"Projects & Portfolio": "Hands-on mini projects, case studies, and professional showcase work.",
	}
	resource_types = {
		"PDF Notes": "Downloadable notes designed for focused study and quick revision.",
		"YouTube Video": "Video-based learning content for conceptual understanding and demos.",
		"Documentation": "Official or curated references for deep learning and documentation review.",
		"Article": "Short-form explainers and practical workflow guides.",
		"Code Example": "Reusable code snippets and implementation patterns.",
		"Cheat Sheet": "Condensed summaries for tool usage and fast recall.",
		"Interview Notes": "Practice-focused material for technical and behavioral interviews.",
		"Mini Project": "Portfolio-worthy project ideas and implementation steps.",
		"Dataset": "Structured data to practice on real tasks and analytics challenges.",
	}
	with connection() as database:
		for category_name, description in resource_categories.items():
			database.execute(
				"INSERT OR IGNORE INTO resource_categories (category_name, description) VALUES (?, ?)",
				(category_name, description),
			)
		for type_name, description in resource_types.items():
			database.execute(
				"INSERT OR IGNORE INTO resource_types (type_name, description) VALUES (?, ?)",
				(type_name, description),
			)
		category_map = {item["category_name"]: item["id"] for item in database.execute("SELECT id, category_name FROM resource_categories").fetchall()}
		type_map = {item["type_name"]: item["id"] for item in database.execute("SELECT id, type_name FROM resource_types").fetchall()}
		resources = [
			{
				"category": "Python",
				"type": "Documentation",
				"title": "Python Official Tutorials",
				"description": "Official Python tutorial for syntax, basics, control flow, modules, and practical coding habits.",
				"technology": "Python",
				"topic": "Core Python",
				"difficulty": "Beginner",
				"url": "https://docs.python.org/3/tutorial/",
				"source": "Python.org",
			},
			{
				"category": "Python",
				"type": "YouTube Video",
				"title": "Corey Schafer Python Playlist",
				"description": "Beginner-to-advanced Python videos covering OOP, decorators, file handling, and project patterns.",
				"technology": "Python",
				"topic": "Python practice",
				"difficulty": "Beginner",
				"url": "https://www.youtube.com/@coreyms",
				"source": "YouTube",
			},
			{
				"category": "SQL & Databases",
				"type": "Documentation",
				"title": "W3Schools SQL Tutorial",
				"description": "Learn SELECT, JOIN, aggregations, subqueries, and real database operations with examples.",
				"technology": "SQL",
				"topic": "Database queries",
				"difficulty": "Beginner",
				"url": "https://www.w3schools.com/sql/",
				"source": "W3Schools",
			},
			{
				"category": "Web Development",
				"type": "Documentation",
				"title": "MDN Web Docs",
				"description": "Complete reference for HTML, CSS, JavaScript, accessibility, and responsive web design.",
				"technology": "HTML, CSS, JavaScript",
				"topic": "Frontend basics",
				"difficulty": "Beginner",
				"url": "https://developer.mozilla.org/en-US/docs/Web",
				"source": "MDN",
			},
			{
				"category": "Web Development",
				"type": "YouTube Video",
				"title": "Traversy Media Web Dev Tutorials",
				"description": "Modern frontend and full-stack project tutorials for real-world web development practice.",
				"technology": "HTML, CSS, JavaScript, React",
				"topic": "Frontend projects",
				"difficulty": "Intermediate",
				"url": "https://www.youtube.com/@TraversyMedia",
				"source": "YouTube",
			},
			{
				"category": "Web Development",
				"type": "Code Example",
				"title": "JavaScript Practice Patterns",
				"description": "Explore functions, DOM interactions, event handling, and reusable frontend logic.",
				"technology": "JavaScript",
				"topic": "Frontend scripting",
				"difficulty": "Intermediate",
				"url": "https://www.w3schools.com/js/",
				"source": "W3Schools",
			},
			{
				"category": "Data Analytics",
				"type": "Dataset",
				"title": "Kaggle Explore Datasets",
				"description": "Find real-world data for sales, healthcare, e-commerce, and business analytics projects.",
				"technology": "Excel, SQL, Python",
				"topic": "Data analysis",
				"difficulty": "Intermediate",
				"url": "https://www.kaggle.com/datasets",
				"source": "Kaggle",
			},
			{
				"category": "Data Analytics",
				"type": "Mini Project",
				"title": "Business Dashboard Challenge",
				"description": "Build a sales dashboard with KPIs, trends, filters, and stakeholder-friendly visual summaries.",
				"technology": "Python, SQL, Power BI",
				"topic": "Dashboard project",
				"difficulty": "Intermediate",
				"url": "https://www.kaggle.com/competitions",
				"source": "Kaggle",
			},
			{
				"category": "AI & ML",
				"type": "Documentation",
				"title": "Scikit-Learn User Guide",
				"description": "Strong practical reference for feature engineering, models, metrics, and ML pipelines.",
				"technology": "Machine Learning, Python",
				"topic": "ML foundations",
				"difficulty": "Intermediate",
				"url": "https://scikit-learn.org/stable/user_guide.html",
				"source": "scikit-learn",
			},
			{
				"category": "AI & ML",
				"type": "Article",
				"title": "AI Concepts Cheat Sheet",
				"description": "Compare supervised learning, unsupervised learning, prompt design, and decision pipelines.",
				"technology": "AI, ML",
				"topic": "AI foundations",
				"difficulty": "Intermediate",
				"url": "https://www.w3schools.com/ai/",
				"source": "W3Schools",
			},
			{
				"category": "Cybersecurity",
				"type": "Documentation",
				"title": "Cybersecurity Fundamentals Overview",
				"description": "Understand threats, authentication, access control, and secure-by-design principles.",
				"technology": "Cybersecurity",
				"topic": "Security basics",
				"difficulty": "Beginner",
				"url": "https://www.w3schools.com/cybersecurity/",
				"source": "W3Schools",
			},
			{
				"category": "Cybersecurity",
				"type": "Cheat Sheet",
				"title": "Security Checklist",
				"description": "A quick review of OWASP-aligned posture, password hygiene, patching, and safe deployment practices.",
				"technology": "Cybersecurity, Risk",
				"topic": "Risk and security",
				"difficulty": "Beginner",
				"url": "https://owasp.org/",
				"source": "OWASP",
			},
			{
				"category": "Cloud & DevOps",
				"type": "YouTube Video",
				"title": "DevOps Fundamentals Walkthrough",
				"description": "A practical guide to CI/CD, containers, automation, and deployment pipelines.",
				"technology": "Docker, GitHub Actions",
				"topic": "DevOps workflow",
				"difficulty": "Intermediate",
				"url": "https://www.youtube.com/results?search_query=devops+fundamentals+for+beginners",
				"source": "YouTube",
			},
			{
				"category": "Cloud & DevOps",
				"type": "Documentation",
				"title": "GitHub Docs",
				"description": "Learn repositories, branching, pull requests, workflows, automation, and collaborative dev practices.",
				"technology": "Git, GitHub, CI/CD",
				"topic": "Workflow automation",
				"difficulty": "Intermediate",
				"url": "https://docs.github.com/en",
				"source": "GitHub",
			},
			{
				"category": "Interview Preparation",
				"type": "Interview Notes",
				"title": "InterviewBit Practice Roadmap",
				"description": "Technical interview preparation with patterns, explanations, and problem-solving sets for software roles.",
				"technology": "Coding, Data Structures, Interviewing",
				"topic": "Technical prep",
				"difficulty": "Intermediate",
				"url": "https://www.interviewbit.com/",
				"source": "InterviewBit",
			},
			{
				"category": "Interview Preparation",
				"type": "Code Example",
				"title": "LeetCode Problem Patterns",
				"description": "Practice core algorithm and data structure topics using real interview-style problem sets.",
				"technology": "DSA, Problem Solving",
				"topic": "Interview coding",
				"difficulty": "Intermediate",
				"url": "https://leetcode.com/problemset/",
				"source": "LeetCode",
			},
			{
				"category": "Projects & Portfolio",
				"type": "Mini Project",
				"title": "Portfolio Analytics Dashboard",
				"description": "Create a small dashboard project that showcases your ability to visualize outcomes and trends.",
				"technology": "Python, SQL, Streamlit",
				"topic": "Portfolio project",
				"difficulty": "Intermediate",
				"url": "https://streamlit.io/",
				"source": "Streamlit",
			},
			{
				"category": "Projects & Portfolio",
				"type": "Article",
				"title": "GitHub README and Project Showcase Guide",
				"description": "Learn how to present your technical work clearly with a strong README, architecture notes, and impact statements.",
				"technology": "GitHub, Portfolio",
				"topic": "Project presentation",
				"difficulty": "Beginner",
				"url": "https://github.com/",
				"source": "GitHub",
			},
		]
		for item in resources:
			category_id = category_map.get(item["category"])
			type_id = type_map.get(item["type"])
			if category_id is None or type_id is None:
				continue
			resource_row = database.execute(
				"SELECT id FROM resources WHERE title = ? AND technology = ? AND topic = ?",
				(item["title"], item["technology"], item["topic"]),
			).fetchone()
			if resource_row is None:
				database.execute(
				"INSERT INTO resources (type_id, category_id, title, description, technology, topic, difficulty, url, source, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
				(type_id, category_id, item["title"], item["description"], item["technology"], item["topic"], item["difficulty"], item["url"], item["source"]),
			)
			resource_ref = database.execute(
				"SELECT id FROM resources WHERE title = ? AND technology = ? AND topic = ?",
				(item["title"], item["technology"], item["topic"]),
			).fetchone()
			if resource_ref:
				database.execute(
				"INSERT OR IGNORE INTO resource_metadata (resource_id, meta_key, meta_value) VALUES (?, ?, ?)",
				(resource_ref["id"], "resource_type", item["type"]),
				)
				database.execute(
				"INSERT OR IGNORE INTO resource_metadata (resource_id, meta_key, meta_value) VALUES (?, ?, ?)",
				(resource_ref["id"], "difficulty", item["difficulty"]),
				)


def list_resource_categories() -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute("SELECT id, category_name AS name, description FROM resource_categories ORDER BY id").fetchall()
		return [dict(row) for row in rows]


def list_resource_types() -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute("SELECT id, type_name AS type_name, description FROM resource_types ORDER BY id").fetchall()
		return [dict(row) for row in rows]


def get_resources_for_user(user_id: int, category_name: str | None = None, type_name: str | None = None, search_term: str = "") -> list[dict[str, Any]]:
	query = """
		SELECT r.id, r.title, r.description, r.technology, r.topic, r.difficulty, r.url, r.file_path, r.source,
			rc.category_name, rt.type_name,
			rb.id AS bookmark_id,
			rp.status AS status, rp.progress_percent AS progress_percent
		FROM resources r
		JOIN resource_categories rc ON rc.id = r.category_id
		JOIN resource_types rt ON rt.id = r.type_id
		LEFT JOIN resource_bookmarks rb ON rb.user_id = ? AND rb.resource_id = r.id
		LEFT JOIN resource_progress rp ON rp.user_id = ? AND rp.resource_id = r.id
		WHERE r.active = 1
	"""
	params: list[Any] = [user_id, user_id]
	if category_name and category_name != "All categories":
		query += " AND rc.category_name = ?"
		params.append(category_name)
	if type_name and type_name != "All types":
		query += " AND rt.type_name = ?"
		params.append(type_name)
	if search_term and search_term.strip():
		normalized = f"%{search_term.strip().lower()}%"
		query += " AND (LOWER(r.title) LIKE ? OR LOWER(r.description) LIKE ? OR LOWER(r.technology) LIKE ? OR LOWER(r.topic) LIKE ? OR LOWER(rc.category_name) LIKE ? OR LOWER(rt.type_name) LIKE ?)"
		params.extend([normalized] * 6)
	query += " ORDER BY rc.category_name, r.title"
	with connection() as database:
		rows = database.execute(query, params).fetchall()
		results: list[dict[str, Any]] = []
		for row in rows:
			entry = dict(row)
			entry["status"] = entry.get("status") or "Not Started"
			entry["progress_percent"] = int(entry.get("progress_percent") or 0)
			entry["is_bookmarked"] = bool(entry.get("bookmark_id"))
			results.append(entry)
		return results


def toggle_resource_bookmark(user_id: int, resource_id: int) -> bool:
	with connection() as database:
		existing = database.execute("SELECT id FROM resource_bookmarks WHERE user_id = ? AND resource_id = ?", (user_id, resource_id)).fetchone()
		if existing:
			database.execute("DELETE FROM resource_bookmarks WHERE id = ?", (existing["id"],))
			return False
		database.execute("INSERT INTO resource_bookmarks (user_id, resource_id) VALUES (?, ?)", (user_id, resource_id))
		return True


def set_resource_progress(user_id: int, resource_id: int, status: str, progress_percent: int) -> None:
	status_name = (status or "Not Started").strip() or "Not Started"
	progress_value = max(0, min(100, int(progress_percent or 0)))
	with connection() as database:
		database.execute(
			"INSERT INTO resource_progress (user_id, resource_id, status, progress_percent, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(user_id, resource_id) DO UPDATE SET status = excluded.status, progress_percent = excluded.progress_percent, updated_at = CURRENT_TIMESTAMP",
			(user_id, resource_id, status_name, progress_value),
		)


def get_resource_summary(user_id: int) -> dict[str, int]:
	with connection() as database:
		total = database.execute("SELECT COUNT(*) AS count FROM resources WHERE active = 1").fetchone()["count"]
		bookmarked = database.execute("SELECT COUNT(*) AS count FROM resource_bookmarks WHERE user_id = ?", (user_id,)).fetchone()["count"]
		completed = database.execute("SELECT COUNT(*) AS count FROM resource_progress WHERE user_id = ? AND status = 'Completed'", (user_id,)).fetchone()["count"]
		in_progress = database.execute("SELECT COUNT(*) AS count FROM resource_progress WHERE user_id = ? AND status = 'In Progress'", (user_id,)).fetchone()["count"]
		return {"total": int(total), "bookmarked": int(bookmarked), "completed": int(completed), "in_progress": int(in_progress)}


AI_MODE_OPTIONS = ["Chatbot", "Doubt Solver", "Topic Explainer", "Career Guidance", "Study Planner", "Personalized Suggestions", "AI Mentor"]


def get_ai_setting(name: str) -> str:
	try:
		secret = st.secrets.get(name, "")
	except StreamlitSecretNotFoundError:
		secret = ""
	value = str(secret).strip()
	return value or os.getenv(name, "").strip()


def get_ai_provider() -> str:
	if get_ai_setting("GROQ_API_KEY"):
		return "groq"
	if get_ai_setting("OPENAI_API_KEY"):
		return "openai"
	return "local"


def get_ai_model() -> str:
	provider = get_ai_provider()
	if provider == "groq":
		model = get_ai_setting("GROQ_MODEL")
		if not model or model.casefold() in {
			"grok",
			"groq",
			"llama-3.1-8b-instant",
		}:
			return "openai/gpt-oss-20b"
		return model
	if provider == "openai":
		return get_ai_setting("OPENAI_MODEL") or "gpt-4o-mini"
	return "offline-local"


def get_current_ai_conversation(user_id: int, conversation_id: int | None = None) -> dict[str, Any] | None:
	with connection() as database:
		if conversation_id is not None:
			row = database.execute("SELECT * FROM ai_conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)).fetchone()
		else:
			row = database.execute("SELECT * FROM ai_conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
		return dict(row) if row else None


def create_ai_conversation(user_id: int, mode: str = "Chatbot", title: str | None = None) -> int:
	with connection() as database:
		normalized_title = title or f"{mode} conversation"
		cursor = database.execute(
			"INSERT INTO ai_conversations (user_id, title, mode) VALUES (?, ?, ?)",
			(user_id, normalized_title, mode),
		)
		return int(cursor.lastrowid)


def update_ai_conversation_mode(conversation_id: int, user_id: int, mode: str) -> None:
	with connection() as database:
		database.execute(
			"UPDATE ai_conversations SET mode = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
			(mode, conversation_id, user_id),
		)


def save_ai_message(conversation_id: int, user_id: int, role: str, content: str) -> None:
	with connection() as database:
		database.execute(
			"INSERT INTO ai_messages (conversation_id, role, content) VALUES (?, ?, ?)",
			(conversation_id, role, content),
		)
		database.execute(
			"UPDATE ai_conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
			(conversation_id, user_id),
		)


def list_ai_conversations(user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute(
			"SELECT * FROM ai_conversations WHERE user_id = ? ORDER BY updated_at DESC, id DESC LIMIT 10",
			(user_id,),
		).fetchall()
		return [dict(row) for row in rows]


def list_ai_messages(conversation_id: int, user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		row = database.execute("SELECT id FROM ai_conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)).fetchone()
		if row is None:
			return []
		rows = database.execute(
			"SELECT role, content, created_at FROM ai_messages WHERE conversation_id = ? ORDER BY id ASC",
			(conversation_id,),
		).fetchall()
		return [dict(r) for r in rows]


def build_ai_profile_summary(user_id: int) -> dict[str, Any]:
	profile = get_user_profile(user_id)
	analysis = get_analysis(user_id)
	recommendations = get_recommendations_for_user(user_id)
	goal = profile.get("career_goal") or "Not specified"
	skills = profile.get("skills") or []
	if isinstance(skills, str):
		skills = [item.strip() for item in skills.split(",") if item.strip()]
	resume_summary = {} if not analysis else analysis.get("data", {})
	best_match = recommendations[0] if recommendations else {}
	return {
		"career_goal": goal,
		"skills": skills,
		"education": profile.get("education") or "Not provided",
		"experience_summary": profile.get("experience_summary") or "Not provided",
		"current_strengths": profile.get("skills") or ["No skills registered yet"],
		"resume_keywords": resume_summary.get("keywords", [])[:6],
		"best_career": best_match.get("career_name", "No recommendation available"),
		"missing_skills": best_match.get("missing_skills", [])[:5],
	}


def build_local_ai_response(user_id: int, mode: str, question: str) -> str:
	context = build_ai_profile_summary(user_id)
	goal = context["career_goal"]
	skills = ", ".join(context["skills"][:6]) if context["skills"] else "No skills registered yet"
	missing = ", ".join(context["missing_skills"][:4]) if context["missing_skills"] else "core fundamentals and project experience"
	question_text = question.strip()
	if mode == "Doubt Solver":
		return (
			f"Here is a beginner-friendly explanation for: '{question_text}'\n\n"
			"Explanation: Start with the core idea first. Break the topic into simple parts and connect it to a real-world problem so the concept is easier to remember.\n\n"
			"Key points:\n- Focus on the main idea before the details.\n- Learn the terminology and the common use case.\n- Practice with a small example and compare it with a similar concept.\n\n"
			"Example: If you are learning Python OOP, think of a class as a blueprint and an object as an actual instance built from that blueprint.\n\n"
			"Common mistakes:\n- Memorizing definitions without understanding their purpose.\n- Learning too many advanced details before the basics are clear.\n\n"
			"Suggested next topic: Try the same concept in a short mini-project so you can retain it through practice."
		)
	if mode == "Topic Explainer":
		return (
			f"Topic: {question_text or 'CareerAI learning concept'}\n\n"
			"Introduction: This topic matters because it builds a core skill you can reuse in projects and interviews.\n\n"
			"Simple explanation: Think of this as a practical concept with a clear purpose, not just an abstract term.\n\n"
			"Key concepts:\n- Understand the definition.\n- Learn how it is used in real development.\n- Practice one example and explain it in your own words.\n\n"
			"Practical use: Use this concept in a small project, assignment, or coding task that matches your current aim.\n\n"
			"Interview question: 'How would you explain this topic in one sentence to a beginner?'\n\n"
			"Practice suggestion: Solve a mini exercise and write a 3-step summary of the concept."
		)
	if mode == "Career Guidance":
		return (
			f"Career guidance for your current target: {goal}\n\n"
			f"Current strengths: You have skills in {skills}.\n\n"
			f"Skill gaps to address: {missing}.\n\n"
			"Recommended next steps:\n- Build one project aligned to your target role.\n- Focus on a small set of missing skills, not every technology at once.\n- Review learning topics in the Learning Hub that match your roadmap.\n\n"
			"Interview prep: Practice explaining your projects, your choices, and the trade-offs you made.\n\n"
			"If any information is missing, the system should say so rather than inventing details."
		)
	if mode == "Study Planner":
		return (
			f"Study plan for {goal}\n\n"
			"Week 1: Build basics and strengthen your main technical foundation.\n"
			"Week 2: Practice one core skill through a mini project or coding exercise.\n"
			"Week 3: Work on missing areas using structured resources and guided tasks.\n"
			"Week 4: Review, revise, and complete a portfolio-ready task aligned to your career target.\n\n"
			"Daily structure: 30-40 minutes on concept review, 20-30 minutes of practice, and 10-20 minutes for reflection or notes."
		)
	if mode == "Personalized Suggestions":
		return (
			f"Recommended for you based on your profile and current goal: {goal}\n\n"
			"1. Improve the most important missing skill in your current roadmap.\n"
			"2. Build a small project that clearly demonstrates your target role.\n"
			"3. Review one focused resource from the Learning Hub or Resources section.\n"
			"4. Practice an interview question connected to your current skill gap.\n\n"
			"Next best action: choose one skill gap and complete a focused exercise within the next 48 hours."
		)
	if mode == "AI Mentor":
		return (
			f"AI Mentor guidance for {goal}\n\n"
			"Your current direction is promising. Keep your learning focused on the next gap, not every new tool.\n\n"
			"Priority focus:\n- Strengthen the most important missing technical skill.\n- Build one project that proves business value.\n- Improve your resume and interview readiness with measurable outcomes.\n\n"
			"Long-term plan: keep a learning streak, add projects, and review progress every 2-4 weeks before moving to the next milestone."
		)
	return (
		f"Based on your CareerAI profile, your goal is {goal}. Your current skills include {skills}.\n\n"
		"Try focusing on the most relevant technical gap and pair it with a small project or practice exercise to improve retention and confidence."
	)


def call_llm_for_ai_response(user_id: int, mode: str, user_prompt: str) -> str:
	context = build_ai_profile_summary(user_id)
	provider = get_ai_provider()
	if provider != "local":
		with connection() as database:
			service = database.execute(
				"SELECT enabled FROM ai_model_registry WHERE service_key = 'ai_assistant'"
			).fetchone()
		if service and not service["enabled"]:
			provider = "local"
	if provider == "local":
		return build_local_ai_response(user_id, mode, user_prompt)
	model_name = get_ai_model()
	system_prompt = (
		"You are CareerAI's AI assistant. Help students and job seekers with career development, resume guidance, learning explanations, and planning. "
		"Use the user context carefully, avoid inventing personal facts, and clearly say when information is unavailable."
	)
	messages = [
		{"role": "system", "content": system_prompt},
		{
			"role": "user",
			"content": json.dumps({
				"mode": mode,
				"career_goal": context.get("career_goal", "Not specified"),
				"skills": context.get("skills", []),
				"missing_skills": context.get("missing_skills", []),
				"question": user_prompt,
			}, ensure_ascii=True),
		},
	]
	endpoint = "https://api.openai.com/v1/chat/completions" if provider == "openai" else "https://api.groq.com/openai/v1/chat/completions"
	api_key = get_ai_setting("GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY")
	if not api_key:
		return build_local_ai_response(user_id, mode, user_prompt)
	payload = json.dumps({
		"model": model_name,
		"messages": messages,
		"temperature": 0.6,
		"max_tokens": 600,
	}).encode("utf-8")
	request = urllib_request.Request(
		endpoint,
		data=payload,
		headers={
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}",
			"User-Agent": "CareerAI/1.0",
		},
		method="POST",
	)
	try:
		with urllib_request.urlopen(request, timeout=20) as response:
			response_data = json.loads(response.read().decode("utf-8"))
			answer = response_data["choices"][0]["message"]["content"].strip()
			if not answer:
				raise ValueError("The AI provider returned an empty response.")
			return answer
	except urllib_error.HTTPError as error:
		provider_message = ""
		try:
			error_body = json.loads(error.read().decode("utf-8"))
			provider_message = str(error_body.get("error", {}).get("message", ""))[:300]
			provider_code = str(error_body.get("error", {}).get("code", ""))
		except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
			provider_message = ""
			provider_code = ""
		if error.code == 401:
			return "The AI provider rejected the API key (HTTP 401). Check that the key is active and copied correctly."
		if error.code == 403 and provider_code == "1010":
			return "The AI provider or network security layer blocked this request (HTTP 403, code 1010). This response does not confirm whether the API key is valid. Try again from another network or contact Groq support."
		detail = f" Provider detail: {provider_message}" if provider_message else ""
		return f"The AI provider returned HTTP {error.code}.{detail}"
	except (urllib_error.URLError, TimeoutError) as error:
		return f"Could not reach the AI provider: {str(error.reason if isinstance(error, urllib_error.URLError) else error)[:200]}"
	except Exception as error:
		return f"AI response failed: {type(error).__name__}: {str(error)[:200]}"


def build_ai_keyword_suggestions(user_id: int) -> list[dict[str, str]]:
	profile = get_user_profile(user_id)
	goal = profile.get("career_goal") or "Career development"
	skills = profile.get("skills") or []
	items = [
		{"title": "Strengthen your most relevant skill gap", "details": "Choose the skill that most directly blocks your target role and work on it with a short project.", "action_label": "Review skill gap", "resource_url": "#"},
		{"title": "Update your resume with measurable outcomes", "details": "Add impact-driven examples to your projects and experience sections to strengthen applications.", "action_label": "Improve resume", "resource_url": "#"},
		{"title": f"Focus on {goal}", "details": "Align your next learning block with the skill requirements for your chosen career path.", "action_label": "Plan next step", "resource_url": "#"},
	]
	if skills:
		items.insert(0, {"title": "Use your current strengths", "details": f"Your profile already shows strength in {', '.join(skills[:4])}. Build a project that makes those strengths visible.", "action_label": "Create project", "resource_url": "#"})
	return items


def save_ai_suggestions(user_id: int) -> list[dict[str, str]]:
	items = build_ai_keyword_suggestions(user_id)
	with connection() as database:
		for item in items:
			database.execute(
				"INSERT INTO ai_suggestions (user_id, category, title, details, action_label, resource_url) VALUES (?, 'Recommended', ?, ?, ?, ?)",
				(user_id, item["title"], item["details"], item["action_label"], item["resource_url"]),
			)
	return items


def get_ai_suggestions(user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute("SELECT * FROM ai_suggestions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user_id,)).fetchall()
		if rows:
			return [dict(row) for row in rows]
	return save_ai_suggestions(user_id)


def create_study_plan_from_profile(user_id: int, daily_minutes: int = 60, weekly_days: int = 5, target_weeks: int = 4) -> tuple[str, list[dict[str, Any]]]:
	profile = get_user_profile(user_id)
	goal = profile.get("career_goal") or "Career development"
	skills = profile.get("skills") or []
	if isinstance(skills, str):
		skills = [item.strip() for item in skills.split(",") if item.strip()]
	recommendations = get_recommendations_for_user(user_id)
	missing = []
	if recommendations:
		missing = recommendations[0].get("missing_skills", [])[:4]
	order = [
		("Week 1", "Foundations", "Learning Hub or project resource", "Solve a short guided exercise", "Review definitions and practice basic syntax"),
		("Week 2", "Core skill practice", "Use a practical resource suited to your target role", "Code a mini project", "Build one focused project deliverable"),
		("Week 3", "Gap closure", "Focus on your most important missing skill", "Complete a targeted challenge", "Implement a skill-based assignment"),
		("Week 4", "Review and apply", "Use a final project or revision resource", "Prepare interview answers or portfolio evidence", "Document your learning and outcomes"),
	]
	tasks = []
	for idx, (day_label, topic, resource, practice, task) in enumerate(order, start=1):
		topic_name = missing[idx - 1] if idx - 1 < len(missing) else (skills[idx - 1] if idx - 1 < len(skills) else "Core skill reinforcement")
		tasks.append({
			"day_label": day_label,
			"topic": topic_name,
			"resource": resource,
			"practice": practice,
			"task": task,
			"estimated_minutes": daily_minutes,
			"completion_status": "Pending",
			"sort_order": idx,
		})
	plan_title = f"{goal} study plan"
	with connection() as database:
		cursor = database.execute(
			"INSERT INTO study_plans (user_id, title, target_career, daily_minutes, weekly_days, target_weeks) VALUES (?, ?, ?, ?, ?, ?)",
			(user_id, plan_title, goal, daily_minutes, weekly_days, target_weeks),
		)
		plan_id = int(cursor.lastrowid)
		for task in tasks:
			database.execute(
				"INSERT INTO study_plan_tasks (plan_id, day_label, topic, resource, practice, task, estimated_minutes, completion_status, sort_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(plan_id, task["day_label"], task["topic"], task["resource"], task["practice"], task["task"], task["estimated_minutes"], task["completion_status"], task["sort_order"]),
			)
	return plan_title, tasks


def list_user_study_plans(user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		plans = database.execute("SELECT * FROM study_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user_id,)).fetchall()
		result: list[dict[str, Any]] = []
		for plan in plans:
			tasks = database.execute(
				"SELECT * FROM study_plan_tasks WHERE plan_id = ? ORDER BY sort_order ASC",
				(plan["id"],),
			).fetchall()
			result.append({
				"plan": dict(plan),
				"tasks": [dict(task) for task in tasks],
			})
		return result


def save_mentor_session(user_id: int, focus: str, summary: str, recommendation: str) -> None:
	with connection() as database:
		database.execute(
			"INSERT INTO ai_mentor_sessions (user_id, focus, summary, recommendation) VALUES (?, ?, ?, ?)",
			(user_id, focus, summary, recommendation),
		)


def get_mentor_history(user_id: int) -> list[dict[str, Any]]:
	with connection() as database:
		rows = database.execute("SELECT * FROM ai_mentor_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 4", (user_id,)).fetchall()
		return [dict(row) for row in rows]


# TAB: AI Assistant
# This page handles chat modes, mentor guidance, suggestions, and AI-powered support.
def render_ai_assistant_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	st.header("AI Assistant")
	st.caption("CareerAI support for learning, planning, guidance, and targeted action.")
	selected_mode = st.selectbox("Mode selection", AI_MODE_OPTIONS, index=0, key=f"ai_mode_select_{user_id}")
	left_col, right_col = st.columns([0.32, 0.68])
	with left_col:
		st.subheader("Recent conversations")
		if st.button("New chat", key=f"ai_new_chat_{user_id}", use_container_width=True):
			st.session_state.pop(f"ai_conversation_id_{user_id}", None)
			st.rerun()
		for conversation in list_ai_conversations(user_id):
			if st.button(f"{conversation['title']} ({conversation['mode']})", key=f"ai_conv_{conversation['id']}", use_container_width=True):
				st.session_state[f"ai_conversation_id_{user_id}"] = conversation["id"]
				st.session_state[f"ai_mode_{user_id}"] = conversation["mode"]
				st.rerun()
		if not list_ai_conversations(user_id):
			st.caption("No conversations yet. Start one now.")
	with right_col:
		conversation_id = st.session_state.get(f"ai_conversation_id_{user_id}")
		mode_key = st.session_state.get(f"ai_mode_{user_id}", selected_mode)
		if conversation_id is None:
			conversation_id = create_ai_conversation(user_id, mode_key, title=f"{mode_key} session")
			st.session_state[f"ai_conversation_id_{user_id}"] = conversation_id
		update_ai_conversation_mode(conversation_id, user_id, mode_key if mode_key in AI_MODE_OPTIONS else selected_mode)
		if mode_key != selected_mode:
			mode_key = selected_mode
			st.session_state[f"ai_mode_{user_id}"] = mode_key
		st.caption(f"Active mode: {mode_key}")
		messages = list_ai_messages(conversation_id, user_id)
		for message in messages:
			if message["role"] == "user":
				with st.chat_message("user"):
					st.write(message["content"])
			else:
				with st.chat_message("assistant"):
					st.write(message["content"])
		if mode_key == "Study Planner":
			with st.form(f"study_planner_form_{user_id}"):
				daily_minutes = st.number_input("Daily study time (minutes)", min_value=15, max_value=240, value=60, step=15)
				weekly_days = st.number_input("Weekly study days", min_value=1, max_value=7, value=5)
				target_weeks = st.number_input("Target period (weeks)", min_value=2, max_value=12, value=4)
				create_plan = st.form_submit_button("Generate study plan")
			if create_plan:
				plan_title, tasks = create_study_plan_from_profile(user_id, daily_minutes, weekly_days, target_weeks)
				st.success(f"Study plan created: {plan_title}")
				for task in tasks:
					st.markdown(f"**{task['day_label']}** · {task['topic']} · {task['estimated_minutes']} minutes")
					st.caption(f"Resource: {task['resource']} | Practice: {task['practice']} | Task: {task['task']}")
				st.session_state[f"ai_plan_snapshot_{user_id}"] = tasks
		elif mode_key == "Personalized Suggestions":
			suggestions = get_ai_suggestions(user_id)
			for item in suggestions:
				with st.container(border=True):
					st.markdown(f"### {item['title']}")
					st.write(item["details"])
					st.button(item["action_label"], key=f"ai_suggestion_{item['id']}_{user_id}", use_container_width=True)
		else:
			suggested_prompts = {
				"Chatbot": ["What should I learn next for AI Engineer?", "How can I improve my resume for internship applications?", "Can you help me plan my next 30 days?"],
				"Doubt Solver": ["Explain Python OOP.", "What is SQL JOIN?", "How does FastAPI work?"],
				"Topic Explainer": ["Explain Python OOP.", "Explain SQL JOIN.", "Explain Machine Learning regression."],
				"Career Guidance": ["Am I ready for AI Engineer?", "What are my skill gaps for Data Analyst?", "What should I learn next?"],
				"AI Mentor": ["What should my next milestone be?", "How should I plan my growth?", "How should I prepare for interviews?"],
			}.get(mode_key, ["Tell me what I should do next."])
			st.caption("Suggested prompts")
			for prompt in suggested_prompts[:3]:
				if st.button(prompt, key=f"ai_prompt_{mode_key}_{prompt}_{user_id}", use_container_width=True):
					st.session_state[f"ai_prompt_input_{user_id}"] = prompt
					st.rerun()
			if prompt := st.chat_input("Ask CareerAI..."):
				st.session_state[f"ai_prompt_input_{user_id}"] = prompt
			if st.session_state.get(f"ai_prompt_input_{user_id}"):
				prompt = st.session_state.pop(f"ai_prompt_input_{user_id}")
				if prompt.strip():
					save_ai_message(conversation_id, user_id, "user", prompt)
					response = call_llm_for_ai_response(user_id, mode_key, prompt)
					save_ai_message(conversation_id, user_id, "assistant", response)
					with st.chat_message("assistant"):
						st.write(response)
					st.rerun()
				else:
					st.caption("Type a question to continue the conversation.")
		if mode_key == "AI Mentor":
			focus = st.selectbox("Mentor focus", ["Career planning", "Skill development", "Project selection", "Resume improvement", "Interview preparation"], key=f"ai_mentor_focus_{user_id}")
			if st.button("Start mentor guidance", key=f"ai_mentor_run_{user_id}"):
				context = build_ai_profile_summary(user_id)
				mentor_summary = (
					f"Your target is {context['career_goal']}. Your current strengths are {', '.join(context['skills'][:4]) if context['skills'] else 'not fully documented'}. "
					f"The most useful next step is to focus on {context['missing_skills'][0] if context['missing_skills'] else 'a project aligned to your target role'} and turn that into a measurable outcome."
				)
				recommendation = "Keep a short learning streak, build one portfolio project, and review progress every two weeks."
				save_mentor_session(user_id, focus, mentor_summary, recommendation)
				st.success("AI Mentor guidance saved and ready to review.")
			mentor_history = get_mentor_history(user_id)
			if mentor_history:
				st.subheader("Recent mentor sessions")
				for item in mentor_history[:3]:
					with st.container(border=True):
						st.markdown(f"**{item['focus']}**")
						st.write(item["summary"])
						st.caption(item["recommendation"])


# ---------------------------
# TAB: Practice & Interview
# ---------------------------

# DATA: Quiz bank
PRACTICE_QUIZ_BANK = {
	"Python": [
		{"question": "Which keyword is used to define a function in Python?", "options": ["func", "def", "lambda", "class"], "answer": "def", "explanation": "Python uses the def keyword to define a function."},
		{"question": "What does list.append() do?", "options": ["Removes the last item", "Adds an item to the end", "Sorts the list", "Creates a copy"], "answer": "Adds an item to the end", "explanation": "append() adds a new item to the end of a list."},
	],
	"SQL": [
		{"question": "Which SQL clause filters rows?", "options": ["GROUP BY", "WHERE", "ORDER BY", "JOIN"], "answer": "WHERE", "explanation": "WHERE is used to filter rows before grouping or sorting."},
		{"question": "Which command is used to fetch rows from a table?", "options": ["SELECT", "INSERT", "UPDATE", "DELETE"], "answer": "SELECT", "explanation": "SELECT retrieves rows from one or more tables."},
	],
	"DSA": [
		{"question": "Which data structure uses FIFO order?", "options": ["Stack", "Queue", "Set", "Tree"], "answer": "Queue", "explanation": "Queue follows First In, First Out (FIFO) ordering."},
		{"question": "What is the time complexity of a binary search on a sorted list?", "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "answer": "O(log n)", "explanation": "Binary search halves the search space each step."},
	],
	"Machine Learning": [
		{"question": "What does model accuracy measure?", "options": ["How fast a model runs", "How often predictions are correct", "How large the dataset is", "How many features the model has"], "answer": "How often predictions are correct", "explanation": "Accuracy measures the percentage of correct predictions on a dataset."},
	],
}

# DATA: MCQ test bank
PRACTICE_MCQ_BANK = {
	"Beginner": [
		{"question": "What is a variable in Python?", "options": ["A fixed value", "A named storage location", "A file", "A function"], "answer": "A named storage location", "explanation": "A variable stores data under a meaningful name."},
		{"question": "Which of these is a valid Python list?", "options": ["{1, 2, 3}", "[1, 2, 3]", "(1, 2, 3)", "'1, 2, 3'"], "answer": "[1, 2, 3]", "explanation": "Lists in Python use square brackets."},
	],
	"Intermediate": [
		{"question": "Which keyword lets you handle exceptions in Python?", "options": ["if", "for", "try", "def"], "answer": "try", "explanation": "The try statement is used for exception handling."},
		{"question": "What is the purpose of a dictionary?", "options": ["Store ordered pairs of keys and values", "Store a sequence of numbers only", "Compile code", "Create classes"], "answer": "Store ordered pairs of keys and values", "explanation": "Dictionaries map keys to values for efficient lookup."},
	],
	"Advanced": [
		{"question": "What is a generator in Python?", "options": ["A loop construct only", "A lazy iterable that yields values one at a time", "A database object", "A text editor"], "answer": "A lazy iterable that yields values one at a time", "explanation": "Generators produce values on demand and save memory."},
	],
}

# DATA: Coding practice problems
PRACTICE_CODING_PROBLEMS = [
	{
		"category": "Python",
		"title": "Reverse a string",
		"difficulty": "Beginner",
		"description": "Write a function that reverses a given string without using slicing.",
		"input": "A string like 'careerai'",
		"output": "A reversed string like 'iarecraC'",
		"example": "Input: careerai\nOutput: iarecraC",
		"solution_concept": "Use a loop and build a result string character by character.",
		"topic": "String manipulation",
	},
	{
		"category": "SQL",
		"title": "Find top customers",
		"difficulty": "Intermediate",
		"description": "Write a query to list the top 3 customers by total purchase amount.",
		"input": "orders table with customer_id and total_amount",
		"output": "Rows sorted by highest total purchase amount",
		"example": "SELECT customer_id, SUM(total_amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 3;",
		"solution_concept": "Use GROUP BY, SUM(), and ORDER BY with LIMIT.",
		"topic": "Aggregation",
	},
	{
		"category": "JavaScript",
		"title": "Check palindrome",
		"difficulty": "Beginner",
		"description": "Create a function that checks whether a word is a palindrome.",
		"input": "A word or sentence",
		"output": "true or false",
		"example": "Input: 'level'\nOutput: true",
		"solution_concept": "Normalize the input and compare it with its reverse.",
		"topic": "Logic and string handling",
	},
]

# DATA: Mock interview questions
PRACTICE_MOCK_INTERVIEW_BANK = {
	"Python Developer": [
		{"question": "Explain the difference between a list and a tuple in Python.", "keywords": ["list", "tuple", "mutable", "immutable"]},
		{"question": "How would you optimize a Python function that processes a large dataset?", "keywords": ["performance", "loop", "optimize", "memory", "pandas"]},
	],
	"AI Engineer": [
		{"question": "What is the difference between training and inference in machine learning?", "keywords": ["training", "inference", "model", "predict"]},
	],
	"Data Scientist": [
		{"question": "How do you evaluate whether a model is good enough for production?", "keywords": ["metrics", "accuracy", "precision", "recall", "validation"]},
	],
}

# DATA: HR interview questions
PRACTICE_HR_QUESTIONS = [
	"Tell me about yourself.",
	"What motivates you to pursue this role?",
	"Describe a challenge you solved at work or in a project.",
]

# DATA: Technical interview questions
PRACTICE_TECHNICAL_QUESTIONS = {
	"Python": [
		"How do you debug a Python program when it fails at runtime?",
		"Explain the difference between a function and a method.",
	],
	"SQL": [
		"How do you write a query to find duplicate rows?",
		"Explain the difference between INNER JOIN and LEFT JOIN.",
	],
	"Web Development": [
		"How would you structure a project using HTML, CSS, and JavaScript?",
		"What are the common performance issues in frontend apps?",
	],
}

# DATA: Company-wise questions
PRACTICE_COMPANY_QUESTIONS = {
	"Google": [
		"Tell me about a project where you improved performance.",
		"How do you write clean, maintainable code?",
	],
	"Microsoft": [
		"How would you handle a conflict in a team?",
		"What is your approach to learning new technologies?",
	],
	"Amazon": [
		"How do you prioritize work when requirements change quickly?",
		"Describe a time you improved a process with measurable impact.",
	],
}


# TAB: Practice & Interview
# This page renders all practice sections: Quiz, MCQ, Coding Practice, Mock Interview, and company prep.
def render_practice_interview_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	st.header("Practice & Interview")
	st.caption("Simple Python-based practice, mock interviews, and job-readiness exercises. No compiler downloads or heavy setup required.")
	# TAB: Quiz
	# TAB: MCQ Tests
	# TAB: Coding Practice
	# TAB: Coding Challenges
	# TAB: AI Mock Interview
	# TAB: HR Interview
	# TAB: Technical Interview
	# TAB: Resume vs Job Match
	# TAB: Company-wise Questions
	quiz_tab, mcq_tab, coding_practice_tab, coding_challenge_tab, mock_interview_tab, hr_interview_tab, technical_interview_tab, resume_match_tab, company_tab = st.tabs([
		"Quiz",
		"MCQ Tests",
		"Coding Practice",
		"Coding Challenges",
		"AI Mock Interview",
		"HR Interview",
		"Technical Interview",
		"Resume vs Job Match",
		"Company-wise Questions",
	])

	# TAB BODY: Quiz
	with quiz_tab:
		st.subheader("Quiz")
		technology = st.selectbox("Technology", options=list(PRACTICE_QUIZ_BANK.keys()), key=f"practice_quiz_technology_{user_id}")
		topic = st.selectbox("Topic", options=[q["question"] for q in PRACTICE_QUIZ_BANK[technology]], key=f"practice_quiz_topic_{user_id}")
		difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"], index=0, key=f"practice_quiz_difficulty_{user_id}")
		question_count = st.slider("Number of questions", 1, min(5, len(PRACTICE_QUIZ_BANK[technology])), 2, key=f"practice_quiz_count_{user_id}")
		questions = PRACTICE_QUIZ_BANK[technology][:question_count]
		user_answers: dict[str, str] = {}
		for index, item in enumerate(questions, start=1):
			with st.container(border=True):
				st.markdown(f"**Q{index}. {item['question']}**")
				answer = st.radio("Choose an option", item["options"], key=f"quiz_q_{user_id}_{index}", horizontal=False)
				user_answers[item["question"]] = answer
		if st.button("Submit quiz", key=f"submit_quiz_{user_id}"):
			score = sum(1 for item in questions if user_answers.get(item["question"]) == item["answer"])
			percentage = round((score / len(questions)) * 100, 1) if questions else 0
			st.success(f"Score: {score}/{len(questions)} ({percentage}%)")
			for item in questions:
				st.write(f"• {item['question']} → {item['answer']} — {item['explanation']}")
			st.session_state.setdefault(f"quiz_history_{user_id}", []).insert(0, {"technology": technology, "score": score, "total": len(questions), "percentage": percentage})
			st.session_state[f"quiz_history_{user_id}"] = st.session_state[f"quiz_history_{user_id}"][:5]
			record_practice_attempt(user_id, "Quiz attempted", technology, score, len(questions))
		if st.session_state.get(f"quiz_history_{user_id}"):
			st.subheader("Attempt history")
			for history in st.session_state[f"quiz_history_{user_id}"]:
				st.caption(f"{history['technology']} → {history['score']}/{history['total']} ({history['percentage']}%)")

	# TAB BODY: MCQ Tests
	with mcq_tab:
		st.subheader("MCQ Tests")
		level = st.selectbox("Test level", ["Beginner", "Intermediate", "Advanced"], key=f"mcq_level_{user_id}")
		test_questions = PRACTICE_MCQ_BANK.get(level, [])
		st.caption("Timer: 10 minutes for this practice test")
		for index, item in enumerate(test_questions, start=1):
			with st.container(border=True):
				st.markdown(f"**Q{index}. {item['question']}**")
				st.radio("Select your answer", item["options"], key=f"mcq_{user_id}_{level}_{index}", horizontal=False)
		if st.button("Submit test", key=f"submit_mcq_{user_id}"):
			score = 0
			for index, item in enumerate(test_questions, start=1):
				selected = st.session_state.get(f"mcq_{user_id}_{level}_{index}")
				if selected == item["answer"]:
					score += 1
			percentage = round((score / len(test_questions)) * 100, 1) if test_questions else 0
			st.success(f"Performance summary: {score}/{len(test_questions)} ({percentage}%)")
			for item in test_questions:
				st.write(f"• {item['answer']} — {item['explanation']}")
			record_practice_attempt(user_id, "MCQ test attempted", level, score, len(test_questions))

	# TAB BODY: Coding Practice
	with coding_practice_tab:
		st.subheader("Coding Practice")
		practice_category = st.selectbox("Technology", ["Python", "SQL", "JavaScript", "DSA", "FastAPI", "Other"], key=f"practice_category_{user_id}")
		filtered_problems = [problem for problem in PRACTICE_CODING_PROBLEMS if problem["category"] == practice_category] or PRACTICE_CODING_PROBLEMS
		selected_problem = st.selectbox("Problem", [problem["title"] for problem in filtered_problems], key=f"coding_problem_{user_id}")
		problem = next(item for item in filtered_problems if item["title"] == selected_problem)
		with st.container(border=True):
			st.markdown(f"### {problem['title']}")
			st.caption(f"{problem['difficulty']} · {problem['topic']}")
			st.write(problem["description"])
			st.markdown(f"**Input:** {problem['input']}")
			st.markdown(f"**Output:** {problem['output']}")
			st.code(problem["example"])
			st.markdown(f"**Expected solution concept:** {problem['solution_concept']}")
		solution_text = st.text_area("Write your solution", height=220, key=f"coding_solution_{user_id}")
		if st.button("Submit solution", key=f"submit_solution_{user_id}"):
			st.info("This version keeps code safe and does not execute arbitrary code on the server. Your draft is stored as a practice submission only.")
			st.session_state.setdefault(f"coding_history_{user_id}", []).insert(0, {"title": problem["title"], "solution": solution_text[:200]})
			st.session_state[f"coding_history_{user_id}"] = st.session_state[f"coding_history_{user_id}"][:5]
			record_practice_attempt(user_id, "Coding solution submitted (not executed)", problem["title"], status="Attempted")
		if st.session_state.get(f"coding_history_{user_id}"):
			st.subheader("Recent submissions")
			for item in st.session_state[f"coding_history_{user_id}"]:
				st.caption(f"{item['title']} — {item['solution'][:80]}")

	# TAB BODY: Coding Challenges
	with coding_challenge_tab:
		st.subheader("Coding Challenges")
		challenge_options = [
			{"title": "Daily Challenge", "difficulty": "Intermediate", "focus": "Python problem solving"},
			{"title": "Recommended Challenge", "difficulty": "Beginner", "focus": "SQL query practice"},
			{"title": "DSA Sprint", "difficulty": "Advanced", "focus": "Array and string logic"},
		]
		challenge = st.selectbox("Challenge", [item["title"] for item in challenge_options], key=f"practice_challenge_{user_id}")
		selected_challenge = next(item for item in challenge_options if item["title"] == challenge)
		with st.container(border=True):
			st.markdown(f"### {selected_challenge['title']}")
			st.caption(f"Difficulty: {selected_challenge['difficulty']}")
			st.write(f"Focus: {selected_challenge['focus']}")
		if st.button("Mark as attempted", key=f"challenge_attempt_{user_id}"):
			st.session_state.setdefault(f"challenge_history_{user_id}", []).insert(0, {"title": challenge, "status": "Attempted"})
			st.session_state[f"challenge_history_{user_id}"] = st.session_state[f"challenge_history_{user_id}"][:5]
			record_practice_attempt(user_id, "Coding challenge attempted", challenge, status="Attempted")
		st.subheader("Challenge History")
		if st.session_state.get(f"challenge_history_{user_id}"):
			for item in st.session_state[f"challenge_history_{user_id}"]:
				st.caption(f"{item['title']} — {item['status']}")
		else:
			st.write("No completed challenge attempts yet.")

	# TAB BODY: AI Mock Interview
	with mock_interview_tab:
		st.subheader("AI Mock Interview")
		job_role = st.selectbox("Career / Job Role", list(PRACTICE_MOCK_INTERVIEW_BANK.keys()), key=f"mock_role_{user_id}")
		interview_type = st.selectbox("Interview Type", ["Technical", "Behavioral", "Case based"], key=f"mock_type_{user_id}")
		difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"], key=f"mock_difficulty_{user_id}")
		question_count = st.slider("Number of questions", 1, min(3, len(PRACTICE_MOCK_INTERVIEW_BANK[job_role])), 2, key=f"mock_count_{user_id}")
		questions = PRACTICE_MOCK_INTERVIEW_BANK[job_role][:question_count]
		current_index = st.session_state.setdefault(f"mock_index_{user_id}", 0)
		if current_index < len(questions):
			item = questions[current_index]
			st.markdown(f"### Question {current_index + 1}")
			st.write(item["question"])
			answer = st.text_area("Your answer", key=f"mock_answer_{user_id}_{current_index}")
			if st.button("Submit answer", key=f"submit_mock_answer_{user_id}_{current_index}"):
				keyword_score = sum(1 for keyword in item["keywords"] if keyword.lower() in answer.lower())
				score = round((keyword_score / max(1, len(item["keywords"]))) * 100, 1)
				scores_key = f"mock_scores_{user_id}"
				st.session_state.setdefault(scores_key, []).append(score)
				st.success(f"Evaluation: {score}/100")
				st.write("Strengths: answer is relevant and mentions core concepts.")
				st.write("Weak areas: add more practical examples and clearer structure.")
				st.session_state[f"mock_index_{user_id}"] = min(current_index + 1, len(questions))
				if current_index + 1 >= len(questions):
					average_score = round(sum(st.session_state[scores_key]) / len(st.session_state[scores_key]), 1)
					record_practice_attempt(user_id, "AI mock interview completed", job_role, average_score, 100)
					st.session_state[scores_key] = []
		else:
			st.success("Interview complete. Review the score and focus on the weak areas listed above.")
			st.session_state[f"mock_index_{user_id}"] = 0

	# TAB BODY: HR Interview
	with hr_interview_tab:
		st.subheader("HR Interview")
		for index, question in enumerate(PRACTICE_HR_QUESTIONS, start=1):
			with st.container(border=True):
				st.markdown(f"**Q{index}. {question}**")
				st.text_area("Your answer", key=f"hr_answer_{user_id}_{index}", height=120)
		if st.button("Generate HR feedback", key=f"hr_feedback_{user_id}"):
			st.info("Keep your answer focused on impact, clarity, and teamwork. Mention a real example and quantify outcomes where possible.")
			record_practice_attempt(user_id, "HR interview practice reviewed", "HR interview", status="Attempted")

	# TAB BODY: Technical Interview
	with technical_interview_tab:
		st.subheader("Technical Interview")
		tech_area = st.selectbox("Technical area", list(PRACTICE_TECHNICAL_QUESTIONS.keys()), key=f"tech_area_{user_id}")
		questions = PRACTICE_TECHNICAL_QUESTIONS[tech_area]
		for index, question in enumerate(questions, start=1):
			with st.container(border=True):
				st.markdown(f"**Q{index}. {question}**")
				st.text_area("Your response", key=f"tech_answer_{user_id}_{tech_area}_{index}", height=120)
		if st.button("Review technical response", key=f"tech_review_{user_id}"):
			st.success("Good answer structure: explain the concept, state the trade-offs, and provide a real-world use case.")
			record_practice_attempt(user_id, "Technical interview practice reviewed", tech_area, status="Attempted")

	# TAB BODY: Resume vs Job Match
	with resume_match_tab:
		st.subheader("Resume vs Job Match")
		profile = get_user_profile(user_id)
		skills = profile.get("skills") or []
		if isinstance(skills, str):
			skills = [skill.strip() for skill in skills.split(",") if skill.strip()]
		job_description = st.text_area("Paste job description", placeholder="Paste a target role description here", key=f"job_match_desc_{user_id}")
		if job_description:
			job_words = {word.lower() for word in re.findall(r"[A-Za-z]+", job_description) if len(word) > 3}
			skill_words = {str(skill).lower() for skill in skills}
			overlap = sorted(skill_words.intersection(job_words))
			match_score = round((len(overlap) / max(1, len(job_words))) * 100, 1)
			st.metric("Match score", f"{match_score}%")
			st.progress(match_score / 100)
			st.write(f"Strong overlap: {', '.join(overlap[:5]) if overlap else 'No direct overlap found yet.'}")
			st.write("Suggested improvements: add role-specific project outcomes, mention technical stack clearly, and align keywords to the target description.")
		else:
			st.info("Paste a job description to compare your profile with the role.")

	# TAB BODY: Company-wise Questions
	with company_tab:
		st.subheader("Company-wise Questions")
		company = st.selectbox("Company", list(PRACTICE_COMPANY_QUESTIONS.keys()), key=f"company_question_{user_id}")
		for index, question in enumerate(PRACTICE_COMPANY_QUESTIONS[company], start=1):
			with st.container(border=True):
				st.markdown(f"**Q{index}. {question}**")
				st.text_area("Prepare your answer", key=f"company_answer_{user_id}_{company}_{index}", height=120)
		if st.button("View company tips", key=f"company_tips_{user_id}"):
			st.info("Answer with STAR structure: Situation, Task, Action, Result. Keep your examples concise and measurable.")


# ---------------------------------------------------------------------------
# TAB: Jobs & Reports
# ---------------------------------------------------------------------------

# DATA: Curated development examples only; these are not live job vacancies.
JOB_ROLE_CATALOG: list[dict[str, Any]] = [
	{"role": "Python Developer", "category": "Python Development", "technology": "Python", "skills": ["Python", "SQL", "Git", "REST APIs", "FastAPI"], "education": ["computer", "software", "engineering", "science"], "level": "Entry level", "location": "Remote", "workplace": "Remote", "preparation": "Build and document a Python API; review SQL, testing, and Git workflows."},
	{"role": "AI/ML Intern", "category": "AI/ML", "technology": "Machine Learning", "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "Statistics"], "education": ["computer", "data", "mathematics", "statistics"], "level": "Internship", "location": "Bengaluru", "workplace": "Hybrid", "preparation": "Complete an end-to-end model project and explain evaluation metrics and limitations."},
	{"role": "Data Scientist", "category": "Data Science", "technology": "Python", "skills": ["Python", "SQL", "Pandas", "Scikit-learn", "Statistics"], "education": ["data", "computer", "statistics", "mathematics"], "level": "Entry level", "location": "Remote", "workplace": "Remote", "preparation": "Practice exploratory analysis, model evaluation, and communicating evidence-based findings."},
	{"role": "Data Analyst", "category": "Data Analytics", "technology": "SQL", "skills": ["SQL", "Microsoft Excel", "Power BI", "Statistics", "Python"], "education": ["data", "business", "computer", "statistics"], "level": "Entry level", "location": "Mumbai", "workplace": "On-site", "preparation": "Create a dashboard project and practice SQL aggregation and concise business summaries."},
	{"role": "Frontend Developer", "category": "Web Development", "technology": "JavaScript", "skills": ["HTML", "CSS", "JavaScript", "React", "Git"], "education": ["computer", "software", "web", "engineering"], "level": "Internship", "location": "Remote", "workplace": "Remote", "preparation": "Publish a responsive, accessible project and explain component and state decisions."},
	{"role": "Backend Developer", "category": "Backend Development", "technology": "Python", "skills": ["Python", "SQL", "REST APIs", "Git", "Docker"], "education": ["computer", "software", "engineering"], "level": "Entry level", "location": "Pune", "workplace": "Hybrid", "preparation": "Build a documented API with validation, database persistence, tests, and clear error handling."},
	{"role": "Full Stack Developer", "category": "Full Stack Development", "technology": "JavaScript", "skills": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL"], "education": ["computer", "software", "engineering"], "level": "Internship", "location": "Hyderabad", "workplace": "Hybrid", "preparation": "Build a small full-stack app and describe the interface-to-database request flow."},
	{"role": "Security Analyst Intern", "category": "Cyber Security", "technology": "Linux", "skills": ["Linux", "Networking", "Security", "Python", "Git"], "education": ["computer", "security", "information", "network"], "level": "Internship", "location": "Delhi", "workplace": "On-site", "preparation": "Review networking and security fundamentals; document a legal, defensive lab project."},
	{"role": "Cloud / DevOps Engineer", "category": "Cloud / DevOps", "technology": "AWS", "skills": ["Linux", "AWS", "Docker", "Kubernetes", "CI/CD"], "education": ["computer", "cloud", "software", "engineering"], "level": "Entry level", "location": "Bengaluru", "workplace": "Hybrid", "preparation": "Deploy a small app and document its configuration, monitoring, and security choices."},
	{"role": "Software Developer Intern", "category": "Software Development", "technology": "Python", "skills": ["Python", "Data Structures", "Git", "Testing", "SQL"], "education": ["computer", "software", "engineering", "science"], "level": "Internship", "location": "Chennai", "workplace": "On-site", "preparation": "Practice data structures, tests, and explaining trade-offs in a portfolio project."},
]

JOB_SAMPLE_LISTINGS: list[dict[str, Any]] = [
	{
		**role,
		"title": f"{role['role']} {'Internship' if kind == 'Internship' else 'Sample Role'}",
		"type": kind,
		"level": "Internship" if kind == "Internship" else role["level"],
		"experience_required": "No prior experience required" if kind == "Internship" else "Relevant project or work experience preferred",
		"is_sample": True,
	}
	for role in JOB_ROLE_CATALOG
	for kind in (("Internship", "Job") if role["level"] != "Internship" else ("Internship",))
]


def get_jobs_profile_data(user_id: int) -> dict[str, Any]:
	profile = get_user_profile(user_id)
	analysis = get_analysis(user_id)
	recommendations = get_recommendations_for_user(user_id)
	with connection() as database:
		learning_row = database.execute(
			"SELECT AVG(progress_percent) AS average FROM learning_user_course_progress WHERE user_id = ?",
			(user_id,),
		).fetchone()
		practice_row = database.execute(
			"SELECT AVG(score * 100.0 / total) AS average FROM practice_attempts WHERE user_id = ? AND score IS NOT NULL AND total > 0",
			(user_id,),
		).fetchone()
	profile_skills = profile.get("skills") or []
	if isinstance(profile_skills, str):
		profile_skills = [value.strip() for value in profile_skills.split(",") if value.strip()]
	skill_values = list(profile_skills)
	if analysis:
		skill_values.extend(row["skill"] for row in analysis.get("skills_list", []) if row.get("source") in {"resume", "profile"})
	normalized_skills = {normalize_skill_name(str(skill)) for skill in skill_values if str(skill).strip()}
	return {
		"profile": profile,
		"analysis": analysis,
		"recommendations": recommendations,
		"skills": sorted({str(skill).strip() for skill in skill_values if str(skill).strip()}, key=str.casefold),
		"normalized_skills": normalized_skills,
		"learning_progress": float(learning_row["average"]) if learning_row and learning_row["average"] is not None else None,
		"practice_average": float(practice_row["average"]) if practice_row and practice_row["average"] is not None else None,
	}


def score_job_match(listing: dict[str, Any], user_data: dict[str, Any]) -> dict[str, Any]:
	profile = user_data["profile"]
	normalized = user_data["normalized_skills"]
	required = listing["skills"]
	matched = [skill for skill in required if normalize_skill_name(skill) in normalized]
	missing = [skill for skill in required if normalize_skill_name(skill) not in normalized]
	skill_component = 100 * len(matched) / max(1, len(required))
	goal = str(profile.get("career_goal") or "").casefold()
	recommendation_names = [str(item.get("career_name", "")).casefold() for item in user_data["recommendations"][:3]]
	role_terms = f"{listing['role']} {listing['category']}".casefold()
	career_component = 100 if goal and any(term in role_terms for term in re.findall(r"[a-z]+", goal) if len(term) > 2) else (
		60 if any(name and (name in role_terms or role_terms in name) for name in recommendation_names) else 0
	)
	education = str(profile.get("education") or "").casefold()
	education_match = bool(education and any(term in education for term in listing["education"]))
	experience = str(profile.get("experience_summary") or "").strip()
	experience_match = listing["type"] == "Internship" or bool(experience)
	components = [
		(skill_component, 50),
		(career_component, 20),
		(100 if education_match else 0, 10),
		(100 if experience_match else 0, 10),
	]
	if user_data["learning_progress"] is not None:
		components.append((user_data["learning_progress"], 5))
	if user_data["practice_average"] is not None:
		components.append((user_data["practice_average"], 5))
	weight_total = sum(weight for _, weight in components)
	score = round(sum(value * weight for value, weight in components) / weight_total)
	return {
		**listing,
		"match_score": score,
		"matched_skills": matched,
		"missing_skills": missing,
		"education_match": education_match,
		"experience_match": experience_match,
		"learning_progress": user_data["learning_progress"],
		"practice_average": user_data["practice_average"],
		"score_breakdown": {
			"skill coverage": round(skill_component),
			"career alignment": round(career_component),
			"education match": 100 if education_match else 0,
			"experience evidence": 100 if experience_match else 0,
			"learning progress": round(user_data["learning_progress"]) if user_data["learning_progress"] is not None else None,
			"scored practice": round(user_data["practice_average"]) if user_data["practice_average"] is not None else None,
		},
		"why_recommended": (
			f"Your profile matches {len(matched)} of {len(required)} required skills"
			+ (f" and aligns with your {profile['career_goal']} career goal." if profile.get("career_goal") else ".")
		),
	}


def get_jobs_report_progress(user_id: int) -> dict[str, Any]:
	with connection() as database:
		courses = database.execute(
			"SELECT COUNT(*) AS total, SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed, AVG(progress_percent) AS average FROM learning_user_course_progress WHERE user_id = ?",
			(user_id,),
		).fetchone()
		learning_rows = database.execute(
			"SELECT date(created_at) AS day FROM learning_activity WHERE user_id = ? GROUP BY date(created_at) ORDER BY day DESC LIMIT 60",
			(user_id,),
		).fetchall()
		learning_trend = database.execute(
			"SELECT date(created_at) AS day, AVG(progress_percent) AS progress FROM learning_activity WHERE user_id = ? AND created_at >= datetime('now', '-30 days') GROUP BY date(created_at) ORDER BY day",
			(user_id,),
		).fetchall()
		practice_rows = database.execute(
			"SELECT kind, subject, score, total, status, created_at FROM practice_attempts WHERE user_id = ? AND score IS NOT NULL AND total > 0 ORDER BY created_at",
			(user_id,),
		).fetchall()
		skill_counts = database.execute(
			"""SELECT a.id, a.created_at, COUNT(es.id) AS skill_count
			   FROM resume_analyses a LEFT JOIN extracted_skills es
			     ON es.analysis_id = a.id AND es.source = 'resume'
			   WHERE a.user_id = ? GROUP BY a.id ORDER BY a.created_at""",
			(user_id,),
		).fetchall()
		recent_activity_rows = database.execute(
			"SELECT created_at FROM activity_history WHERE user_id = ? AND created_at >= datetime('now', '-30 days')",
			(user_id,),
		).fetchall()
		completed_activities = database.execute(
			"""SELECT
				(SELECT COUNT(*) FROM activity_history WHERE user_id = ? AND status = 'Completed') +
				(SELECT COUNT(*) FROM practice_attempts WHERE user_id = ? AND status = 'Completed') +
				(SELECT COUNT(*) FROM learning_activity WHERE user_id = ? AND progress_percent >= 100)
				AS count""",
			(user_id, user_id, user_id),
		).fetchone()["count"]
	learning_days = [str(row["day"]) for row in learning_rows]
	today = datetime.now().date()
	streak = 0
	for offset, day in enumerate(learning_days):
		expected = (today - timedelta(days=offset)).isoformat()
		if day == expected:
			streak += 1
		elif offset == 0 and day == (today - timedelta(days=1)).isoformat():
			streak = 1
		else:
			break
	analysis = get_analysis(user_id)
	practice = [dict(row) for row in practice_rows]
	scored_practice = [float(row["score"]) / float(row["total"]) * 100 for row in practice if row["total"]]
	course_total = int(courses["total"] or 0)
	course_completed = int(courses["completed"] or 0)
	career_data = get_jobs_profile_data(user_id)
	top_recommendation = career_data["recommendations"][0] if career_data["recommendations"] else None
	dimensions: list[tuple[float, float]] = []
	if analysis:
		dimensions.append((float(analysis["overall"]), 0.25))
	if course_total:
		dimensions.append((float(courses["average"] or 0), 0.25))
	if top_recommendation:
		dimensions.append((float(top_recommendation["recommendation_score"]), 0.25))
	if scored_practice:
		dimensions.append((sum(scored_practice) / len(scored_practice), 0.25))
	readiness = round(sum(score * weight for score, weight in dimensions) / sum(weight for _, weight in dimensions)) if dimensions else None
	return {
		"course_total": course_total,
		"course_completed": course_completed,
		"learning_progress": round(float(courses["average"] or 0)),
		"learning_streak": streak,
		"resume_score": int(analysis["overall"]) if analysis else None,
		"career_readiness": readiness,
		"practice_average": round(sum(scored_practice) / len(scored_practice)) if scored_practice else None,
		"practice_attempts": practice,
		"learning_trend": [dict(row) for row in learning_trend],
		"skill_counts": [dict(row) for row in skill_counts],
		"completed_activities": int(completed_activities),
		"recent_activity_rows": [dict(row) for row in recent_activity_rows],
	}


def get_activity_history(user_id: int, start_date: date | None = None, end_date: date | None = None) -> list[dict[str, Any]]:
	activities: list[dict[str, Any]] = []
	with connection() as database:
		rows = database.execute(
			"SELECT activity, module, status, description, created_at FROM activity_history WHERE user_id = ? ORDER BY created_at DESC",
			(user_id,),
		).fetchall()
		activities.extend(dict(row) for row in rows)
		resume_rows = database.execute(
			"""SELECT a.created_at, r.original_filename, s.overall
			   FROM resume_analyses a JOIN resumes r ON r.id = a.resume_id
			   JOIN ats_scores s ON s.analysis_id = a.id
			   WHERE a.user_id = ? ORDER BY a.created_at DESC""",
			(user_id,),
		).fetchall()
		upload_rows = database.execute(
			"SELECT original_filename, uploaded_at FROM resumes WHERE user_id = ? ORDER BY uploaded_at DESC",
			(user_id,),
		).fetchall()
		activities.extend({
			"activity": "Resume analyzed",
			"module": "Resume",
			"status": "Completed",
			"description": f"ATS score {row['overall']}/100 · {row['original_filename']}",
			"created_at": row["created_at"],
		} for row in resume_rows)
		activities.extend({
			"activity": "Resume uploaded",
			"module": "Resume",
			"status": "Completed",
			"description": row["original_filename"],
			"created_at": row["uploaded_at"],
		} for row in upload_rows)
		learning_rows = database.execute(
			"""SELECT course_name, topic_name, progress_percent, created_at
			   FROM learning_activity WHERE user_id = ? ORDER BY created_at DESC""",
			(user_id,),
		).fetchall()
		activities.extend({
			"activity": "Course topic completed" if int(row["progress_percent"]) >= 100 else "Learning progress updated",
			"module": "Learning Hub",
			"status": "Completed" if int(row["progress_percent"]) >= 100 else "In progress",
			"description": f"{row['course_name']} · {row['topic_name']} · {row['progress_percent']}%",
			"created_at": row["created_at"],
		} for row in learning_rows)
		practice_rows = database.execute(
			"SELECT kind, subject, score, total, status, created_at FROM practice_attempts WHERE user_id = ? ORDER BY created_at DESC",
			(user_id,),
		).fetchall()
		activities.extend({
			"activity": row["kind"],
			"module": "Practice & Interview",
			"status": row["status"],
			"description": f"{row['subject']}" + (f" · {row['score']}/{row['total']}" if row["score"] is not None and row["total"] else ""),
			"created_at": row["created_at"],
		} for row in practice_rows)
		recommendation_rows = database.execute(
			"SELECT career_name, recommendation_score, created_at FROM user_career_recommendations WHERE user_id = ? ORDER BY created_at DESC, recommendation_score DESC LIMIT 1",
			(user_id,),
		).fetchall()
		report_rows = database.execute(
			"SELECT created_at FROM generated_reports WHERE user_id = ? ORDER BY created_at DESC",
			(user_id,),
		).fetchall()
		activities.extend({
			"activity": "Career recommendation generated",
			"module": "Career Guidance",
			"status": "Completed",
			"description": f"{row['career_name']} · {row['recommendation_score']}% match",
			"created_at": row["created_at"],
		} for row in recommendation_rows)
		activities.extend({
			"activity": "Resume PDF report generated",
			"module": "Resume",
			"status": "Completed",
			"description": "Saved analysis report",
			"created_at": row["created_at"],
		} for row in report_rows)
	if start_date:
		activities = [item for item in activities if str(item["created_at"])[:10] >= start_date.isoformat()]
	if end_date:
		activities = [item for item in activities if str(item["created_at"])[:10] <= end_date.isoformat()]
	return sorted(activities, key=lambda item: str(item["created_at"]), reverse=True)


def build_jobs_reports_pdf(user: dict[str, Any], report_type: str, sections: list[tuple[str, list[str]]]) -> bytes:
	from io import BytesIO

	buffer = BytesIO()
	document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
	styles = getSampleStyleSheet()
	styles.add(ParagraphStyle(name="JobsReportBrand", parent=styles["Title"], textColor=colors.HexColor("#087E78"), alignment=TA_CENTER, fontSize=20, spaceAfter=6))
	styles.add(ParagraphStyle(name="JobsReportSection", parent=styles["Heading2"], textColor=colors.HexColor("#183B3A"), spaceBefore=12, spaceAfter=5))
	styles.add(ParagraphStyle(name="JobsReportBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=3))
	story = [
		Paragraph("CareerAI", styles["JobsReportBrand"]),
		Paragraph(reportlab_escape(report_type), styles["Heading2"]),
		Paragraph(f"Candidate: {reportlab_escape(str(user.get('full_name', '')))}", styles["JobsReportBody"]),
		Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["JobsReportBody"]),
		Spacer(1, 8),
	]
	for heading, lines in sections:
		story.append(Paragraph(reportlab_escape(heading), styles["JobsReportSection"]))
		for line in lines:
			story.append(Paragraph("• " + reportlab_escape(str(line)), styles["JobsReportBody"]))
	document.build(story)
	return buffer.getvalue()


def build_progress_report_pdf(user: dict[str, Any], user_id: int) -> bytes:
	stats = get_jobs_report_progress(user_id)
	activities = get_activity_history(user_id)[:10]
	return build_jobs_reports_pdf(user, "Progress Report", [
		("Learning", [f"Average course progress: {stats['learning_progress']}%", f"Courses completed: {stats['course_completed']} of {stats['course_total']}", f"Learning streak: {stats['learning_streak']} day(s)"]),
		("Readiness and performance", [
			f"Career readiness: {stats['career_readiness']}%" if stats["career_readiness"] is not None else "Career readiness: Not enough saved data yet",
			f"Latest resume ATS score: {stats['resume_score']}/100" if stats["resume_score"] is not None else "Latest resume ATS score: Not analyzed",
			f"Average scored practice: {stats['practice_average']}%" if stats["practice_average"] is not None else "Scored practice: No recorded scored attempts yet",
			f"Completed activities: {stats['completed_activities']}",
		]),
		("Recent activity", [f"{item['created_at']} · {item['module']} · {item['activity']} — {item['description']}" for item in activities] or ["No activities recorded yet."]),
	])


def build_career_report_data(user_id: int) -> dict[str, Any]:
	data = get_jobs_profile_data(user_id)
	recommendations = data["recommendations"]
	best = recommendations[0] if recommendations else None
	career_name = best["career_name"] if best else str(data["profile"].get("career_goal") or "")
	gaps = skill_gap_analysis(user_id, career_name) if career_name else {}
	return {
		**data,
		"best": best,
		"gaps": gaps,
		"roadmap": build_career_roadmap(user_id, career_name) if career_name else [],
		"learning_path": build_learning_path(user_id, career_name) if career_name else [],
		"progress": get_jobs_report_progress(user_id),
	}


def build_career_report_pdf(user: dict[str, Any], user_id: int) -> bytes:
	report = build_career_report_data(user_id)
	best = report["best"]
	gaps = report["gaps"]
	profile = report["profile"]
	analysis = report["analysis"]
	readiness = report["progress"]["career_readiness"]
	return build_jobs_reports_pdf(user, "Personalized Career Report", [
		("Candidate overview", [
			f"Career goal: {profile.get('career_goal') or 'Not provided'}",
			f"Education: {profile.get('education') or 'Not provided'}",
			f"Experience: {profile.get('experience_summary') or 'Not provided'}",
		]),
		("Current skills", report["skills"] or ["No profile or resume skills recorded yet."]),
		("Recommended career and match", [f"{best['career_name']} · {best['recommendation_score']}% match" if best else "No stored career recommendation is available yet."]),
		("Skill gap", [
			"Matched: " + (", ".join(gaps.get("matching_skills", [])) or "None recorded"),
			"Missing: " + (", ".join(gaps.get("missing_skills", [])) or "No mapped skill gaps"),
		]),
		("Required skills", gaps.get("missing_skills", []) + gaps.get("matching_skills", []) or ["No mapped requirements available."]),
		("Learning roadmap", [f"{stage['stage_name']}: {', '.join(stage['topics'])} · {stage['completion']}% evidence coverage" for stage in report["roadmap"]] or ["No roadmap available for the selected career."]),
		("Recommended learning", [f"{item['topic']} · {item['priority']} priority · {item['status']}" for item in report["learning_path"]] or ["No learning path available yet."]),
		("Practice performance", [
			f"Average scored practice: {report['progress']['practice_average']}%" if report["progress"]["practice_average"] is not None else "No scored practice attempts recorded yet.",
			f"Recorded practice attempts: {len(report['progress']['practice_attempts'])}",
		]),
		("Resume and readiness", [
			f"Latest ATS score: {analysis['overall']}/100" if analysis else "Resume: no analysis available",
			f"Career readiness: {readiness}%" if readiness is not None else "Career readiness: not enough stored data",
		]),
		("Recommended next steps", [
			"Choose one high-priority missing skill and complete a focused Learning Hub topic.",
			"Build a small, documented project demonstrating the recommended role's required skills.",
			"Review the latest resume analysis and update it with accurate, evidence-based outcomes.",
		]),
	])


def read_email_setting(name: str) -> str:
	return os.getenv(name, "").strip()


def email_reports_configured() -> bool:
	return all(read_email_setting(name) for name in ("EMAIL_HOST", "EMAIL_PORT", "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_FROM"))


def send_report_email(user_id: int, recipient: str, report_type: str, pdf_bytes: bytes) -> tuple[bool, str]:
	if not validate_email(recipient):
		return False, "Enter a valid recipient email address."
	if not email_reports_configured():
		message = "Email sending is not configured. Set EMAIL_HOST, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_FROM in the environment."
		with connection() as database:
			database.execute(
				"INSERT INTO email_report_logs (user_id, report_type, recipient, status, message) VALUES (?, ?, ?, 'Not configured', ?)",
				(user_id, report_type, normalize_email(recipient), message),
			)
		return False, message
	try:
		message = EmailMessage()
		message["Subject"] = f"CareerAI {report_type}"
		message["From"] = read_email_setting("EMAIL_FROM")
		message["To"] = normalize_email(recipient)
		message.set_content(f"Your {report_type.lower()} is attached. It was generated by CareerAI for your account.")
		message.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=f"careerai_{report_type.lower().replace(' ', '_')}.pdf")
		port = int(read_email_setting("EMAIL_PORT"))
		smtp_connection = (
			smtplib.SMTP_SSL(read_email_setting("EMAIL_HOST"), port, timeout=20)
			if port == 465 else smtplib.SMTP(read_email_setting("EMAIL_HOST"), port, timeout=20)
		)
		with smtp_connection as smtp:
			if port != 465:
				smtp.starttls()
			smtp.login(read_email_setting("EMAIL_USERNAME"), read_email_setting("EMAIL_PASSWORD"))
			smtp.send_message(message)
	except (ValueError, OSError, smtplib.SMTPException):
		with connection() as database:
			database.execute(
				"INSERT INTO email_report_logs (user_id, report_type, recipient, status, message) VALUES (?, ?, ?, 'Failed', ?)",
				(user_id, report_type, normalize_email(recipient), "Email delivery failed; check server configuration and mail-provider logs."),
			)
		return False, "The email could not be sent. Check the email configuration and provider logs, then try again."
	with connection() as database:
		database.execute(
			"INSERT INTO email_report_logs (user_id, report_type, recipient, status, message) VALUES (?, ?, ?, 'Sent', ?)",
			(user_id, report_type, normalize_email(recipient), "Report email accepted by the configured SMTP server."),
		)
	record_activity(user_id, "Report emailed", "Jobs & Reports", f"{report_type} sent to {normalize_email(recipient)}")
	return True, "Report email sent."


# TAB: Jobs & Reports
def render_jobs_reports_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	user_data = get_jobs_profile_data(user_id)
	all_matches = [score_job_match(listing, user_data) for listing in JOB_SAMPLE_LISTINGS]
	st.header("Jobs & Reports")
	st.caption("Profile-based career matching and reports from your saved CareerAI data. Listings shown here are structured examples, not live or verified vacancies.")
	internship_tab, jobs_tab, activity_tab, progress_tab, resume_tab, career_tab, download_tab, email_tab = st.tabs([
		"Internships", "Jobs", "Activity History", "Progress Reports",
		"Resume Reports", "Career Reports", "Download PDF", "Email Reports",
	])

	# TAB BODY: Internship Recommendation
	with internship_tab:
		st.subheader("Internship Recommendation")
		internships = sorted((item for item in all_matches if item["type"] == "Internship"), key=lambda item: item["match_score"], reverse=True)
		if not user_data["skills"] and not user_data["profile"].get("career_goal"):
			st.info("Add skills or a career goal in My Profile to make these example recommendations more relevant.")
		for item in internships:
			with st.container(border=True):
				st.markdown(f"### {item['category']}")
				st.metric("Profile match", f"{item['match_score']}%")
				st.write(item["why_recommended"])
				st.caption("Example category only · not a live internship opening")
				st.write("**Matching skills:** " + (", ".join(item["matched_skills"]) or "None identified"))
				st.write("**Missing skills:** " + (", ".join(item["missing_skills"]) or "No mapped gaps"))
				st.write("**Required skills:** " + ", ".join(item["skills"]))
				if item["learning_progress"] is not None or item["practice_average"] is not None:
					st.caption(
						"Readiness signals used: "
						+ (f"Learning progress {round(item['learning_progress'])}%" if item["learning_progress"] is not None else "Learning progress not recorded")
						+ " · "
						+ (f"Scored practice {round(item['practice_average'])}%" if item["practice_average"] is not None else "Scored practice not recorded")
					)
				st.write("**Recommended preparation:** " + item["preparation"])
		if st.button("Record internship recommendations viewed", key=f"internship_viewed_{user_id}"):
			record_activity(user_id, "Internship recommendations viewed", "Jobs & Reports", f"Reviewed {len(internships)} example categories.")
			st.success("Activity recorded.")

	# TAB BODY: Job Recommendation
	with jobs_tab:
		st.subheader("Job Recommendation")
		st.caption("Sample role profiles only. CareerAI is not connected to a live job board.")
		st.caption("Match scores use transparent skill (50%), career alignment (20%), education (10%), experience (10%), and available learning/practice evidence (5% each). Missing learning or practice data is excluded rather than scored as zero.")
		filter_columns = st.columns(3)
		with filter_columns[0]:
			role_filter = st.selectbox("Career role", ["All roles"] + sorted({item["role"] for item in all_matches}), key=f"job_role_filter_{user_id}")
			technology_filter = st.selectbox("Technology", ["All technologies"] + sorted({item["technology"] for item in all_matches}), key=f"job_technology_filter_{user_id}")
		with filter_columns[1]:
			level_filter = st.selectbox("Experience level", ["All levels"] + sorted({item["level"] for item in all_matches}), key=f"job_level_filter_{user_id}")
			location_filter = st.selectbox("Location", ["All locations"] + sorted({item["location"] for item in all_matches}), key=f"job_location_filter_{user_id}")
		with filter_columns[2]:
			workplace_filter = st.selectbox("Remote / On-site", ["All workplace types", "Remote", "Hybrid", "On-site"], key=f"job_workplace_filter_{user_id}")
			minimum_match = st.slider("Minimum match score", 0, 100, 0, 5, key=f"job_min_match_{user_id}")
		matches = [
			item for item in all_matches if item["type"] == "Job"
			and (role_filter == "All roles" or item["role"] == role_filter)
			and (technology_filter == "All technologies" or item["technology"] == technology_filter)
			and (level_filter == "All levels" or item["level"] == level_filter)
			and (location_filter == "All locations" or item["location"] == location_filter)
			and (workplace_filter == "All workplace types" or item["workplace"] == workplace_filter)
			and item["match_score"] >= minimum_match
		]
		for item in sorted(matches, key=lambda match: match["match_score"], reverse=True):
			with st.container(border=True):
				left, right = st.columns([3, 1])
				with left:
					st.markdown(f"### {item['role']}")
					st.caption(f"Example role · {item['level']} · {item['location']} · {item['workplace']}")
					st.write(item["why_recommended"])
				with right:
					st.metric("Job match", f"{item['match_score']}%")
				st.write("**Required skills:** " + ", ".join(item["skills"]))
				st.write("**Matched:** " + (", ".join(item["matched_skills"]) or "None yet") + "  |  **Missing:** " + (", ".join(item["missing_skills"]) or "None mapped"))
				st.write(f"**Education evidence:** {'Match found' if item['education_match'] else 'Not found in profile'} · **Experience evidence:** {'Present or not required' if item['experience_match'] else 'Not recorded'}")
				st.write("**Recommended preparation:** " + item["preparation"])
		if not matches:
			st.info("No example roles match those filters. Try lowering the minimum match score or widening the filters.")
		if st.button("Record job recommendations viewed", key=f"job_recommendations_viewed_{user_id}"):
			record_activity(user_id, "Job recommendations viewed", "Jobs & Reports", f"Reviewed {len(matches)} filtered example roles.")
			st.success("Activity recorded.")

	# TAB BODY: Activity History
	with activity_tab:
		st.subheader("Activity History")
		date_range = st.date_input("Date range", value=(datetime.now().date() - timedelta(days=30), datetime.now().date()), key=f"activity_date_range_{user_id}")
		start_date = date_range[0] if isinstance(date_range, tuple) and len(date_range) == 2 else None
		end_date = date_range[1] if isinstance(date_range, tuple) and len(date_range) == 2 else None
		activities = get_activity_history(user_id, start_date, end_date)
		if activities:
			st.dataframe(pd.DataFrame(activities).rename(columns={
				"activity": "Activity", "module": "Module", "created_at": "Date/Time",
				"status": "Status", "description": "Description",
			}), width="stretch", hide_index=True)
		else:
			st.info("No activity is recorded for this date range yet. Resume analyses and learning progress will appear here, along with future practice and report activity.")

	# TAB BODY: Progress Reports
	with progress_tab:
		st.subheader("Progress Reports")
		stats = get_jobs_report_progress(user_id)
		metric_items = [
			("Learning progress", f"{stats['learning_progress']}%"),
			("Courses completed", f"{stats['course_completed']}/{stats['course_total']}"),
			("Quiz / interview practice", f"{stats['practice_average']}%" if stats["practice_average"] is not None else "No scored attempts"),
			("Resume ATS score", f"{stats['resume_score']}/100" if stats["resume_score"] is not None else "Not analyzed"),
			("Career readiness", f"{stats['career_readiness']}%" if stats["career_readiness"] is not None else "Insufficient data"),
			("Learning streak", f"{stats['learning_streak']} day(s)"),
			("Completed activities", str(stats["completed_activities"])),
		]
		metric_cols = st.columns(4)
		for index, (label, value) in enumerate(metric_items):
			with metric_cols[index % len(metric_cols)]:
				st.metric(label, value, border=True)
		left, right = st.columns(2)
		with left:
			st.markdown("**Learning progress history (saved Learning Hub activity)**")
			if stats["learning_trend"]:
				st.line_chart(pd.DataFrame(stats["learning_trend"]).set_index("day")["progress"])
			else:
				st.info("No stored learning activity to chart yet.")
			st.markdown("**Resume skill count by saved analysis**")
			if stats["skill_counts"]:
				st.line_chart(pd.DataFrame(stats["skill_counts"]).set_index("created_at")["skill_count"])
			else:
				st.info("Analyze a resume to begin a skill history.")
		with right:
			st.markdown("**Scored practice attempts**")
			scored = [item for item in stats["practice_attempts"] if item["score"] is not None and item["total"]]
			if scored:
				practice_frame = pd.DataFrame([{
					"Date": item["created_at"],
					"Result %": round(float(item["score"]) / float(item["total"]) * 100, 1),
					"Assessment": f"{item['kind']} · {item['subject']}",
				} for item in scored])
				st.line_chart(practice_frame.set_index("Date")["Result %"])
			else:
				st.info("Quiz and interview scores will appear after you complete a scored practice attempt.")
			st.markdown("**Weekly activity (saved events)**")
			week_activities = get_activity_history(user_id, datetime.now().date() - timedelta(days=6), datetime.now().date())
			if week_activities:
				activity_frame = pd.DataFrame(week_activities)
				activity_frame["Day"] = activity_frame["created_at"].str[:10]
				counts = activity_frame.groupby("Day").size().rename("Activities")
				st.bar_chart(counts)
			else:
				st.info("No activities recorded in the selected recent week.")
		st.caption("Readiness is a transparent summary of available ATS, learning, career-match, and scored practice data. Missing data is not treated as zero.")
		if st.button("Generate progress report activity", key=f"progress_report_generate_{user_id}"):
			record_activity(user_id, "Progress report generated", "Jobs & Reports", "Generated from saved learning, resume, career, and practice records.")
			st.success("Progress report activity saved.")

	# TAB BODY: Resume Reports
	with resume_tab:
		st.subheader("Resume Reports")
		resume_history = list_resume_history(user_id)
		if not resume_history:
			st.info("Analyze a resume in the Resume tab first. This report view reuses that saved analysis.")
		else:
			history_labels = {row["id"]: f"{row['created_at']} · {row['original_filename']} · ATS {row['overall']}/100" for row in resume_history}
			analysis_id = st.selectbox("Saved resume analysis", list(history_labels), format_func=lambda key: history_labels[key], key=f"jobs_resume_history_{user_id}")
			analysis = get_analysis(user_id, int(analysis_id))
			if analysis:
				st.metric("Resume ATS score", f"{analysis['overall']}/100")
				st.write("**Extracted skills:** " + (", ".join(row["skill"] for row in analysis["skills_list"] if row["source"] == "resume") or "None detected"))
				st.write("**Missing skills:** " + (", ".join(row["skill"] for row in analysis["skills_list"] if row["source"] == "missing") or "None mapped"))
				st.write("**Resume strengths:** " + (", ".join(analysis["data"].get("strengths", [])) or "Not available"))
				st.write("**Resume suggestions:**")
				for suggestion in analysis.get("suggestions", []):
					st.markdown(f"- **{suggestion['priority']}:** {suggestion['text']}")
				with st.expander("Section analysis and keyword details"):
					st.write("**Detected sections:** " + (", ".join(analysis["sections"].get("available", [])) or "None detected"))
					st.write("**Missing sections:** " + (", ".join(analysis["sections"].get("missing", [])) or "None detected"))
					st.write("**Keywords:** " + (", ".join(analysis["data"].get("keywords", [])) or "No keywords recorded"))
					st.write("**Score breakdown:** " + ", ".join(f"{key.title()}: {analysis[key]}" for key in ("skills", "keywords", "structure", "education", "experience", "projects", "completeness")))

	# TAB BODY: Career Reports
	with career_tab:
		st.subheader("Personalized Career Report")
		report = build_career_report_data(user_id)
		profile = report["profile"]
		best = report["best"]
		progress = report["progress"]
		overview_col, readiness_col = st.columns(2)
		with overview_col:
			st.markdown("**Candidate overview**")
			st.write(f"Career goal: {profile.get('career_goal') or 'Not provided'}")
			st.write(f"Education: {profile.get('education') or 'Not provided'}")
			st.write(f"Experience: {profile.get('experience_summary') or 'Not provided'}")
		with readiness_col:
			st.metric("Career readiness", f"{progress['career_readiness']}%" if progress["career_readiness"] is not None else "Not enough saved data")
			st.metric("Resume score", f"{report['analysis']['overall']}/100" if report["analysis"] else "Not analyzed")
		st.markdown("**Current skills**")
		st.write(" · ".join(report["skills"]) or "No profile or resume skills recorded yet.")
		if best:
			st.markdown(f"**Recommended career:** {best['career_name']} · {best['recommendation_score']}% match")
			gaps = report["gaps"]
			st.write("**Skill gap:** " + str(gaps.get("skill_gap_percentage", "N/A")) + "%")
			st.write("**Matching skills:** " + (", ".join(gaps.get("matching_skills", [])) or "None"))
			st.write("**Missing skills:** " + (", ".join(gaps.get("missing_skills", [])) or "None mapped"))
			with st.expander("Required skills, roadmap, and recommended learning", expanded=True):
				st.write("**Required skills:** " + (", ".join(best.get("required_skills", [])) or "Not defined"))
				for stage in report["roadmap"]:
					st.markdown(f"**{stage['stage_name']}** · {stage['completion']}% evidence coverage")
					st.caption("Topics: " + ", ".join(stage["topics"]) + " · Outcome: " + stage["expected_outcome"])
				for item in report["learning_path"]:
					st.write(f"• {item['topic']} — {item['priority']} priority · {item['status']}")
		else:
			st.info("Add a career goal and profile/resume evidence to generate a more complete report.")
		st.markdown("**Practice performance**")
		st.write(f"Recorded attempts: {len(progress['practice_attempts'])} · Average scored performance: {progress['practice_average']}%" if progress["practice_average"] is not None else f"Recorded attempts: {len(progress['practice_attempts'])} · No scored performance yet")
		st.markdown("**Recommended next steps**")
		st.write("1. Complete a high-priority missing Learning Hub topic.")
		st.write("2. Demonstrate a required skill in a portfolio project.")
		st.write("3. Revisit your resume analysis and practice answers using evidence from that project.")

	# TAB BODY: Download PDF
	with download_tab:
		st.subheader("Download PDF Reports")
		st.caption("Reports are generated from your saved CareerAI data. Resume PDFs reuse the existing Resume analysis report builder.")
		if st.button("Prepare Progress Report PDF", key=f"prepare_progress_pdf_{user_id}"):
			st.session_state[f"jobs_report_pdf_progress_{user_id}"] = build_progress_report_pdf(user, user_id)
			record_activity(user_id, "Progress report generated", "Jobs & Reports", "Progress report PDF prepared.")
		progress_pdf = st.session_state.get(f"jobs_report_pdf_progress_{user_id}")
		if progress_pdf:
			st.download_button("Download Progress Report PDF", progress_pdf, file_name="careerai_progress_report.pdf", mime="application/pdf", key=f"download_progress_pdf_{user_id}", on_click=record_activity, args=(user_id, "PDF downloaded", "Jobs & Reports", "Progress report PDF downloaded."))
		if st.button("Prepare Resume Report PDF", key=f"prepare_resume_pdf_{user_id}"):
			latest_analysis = get_analysis(user_id)
			if latest_analysis:
				st.session_state[f"jobs_report_pdf_resume_{user_id}"] = generate_resume_report(user, latest_analysis)
				record_activity(user_id, "Resume report generated", "Jobs & Reports", "Resume PDF prepared from the latest saved analysis.")
			else:
				st.warning("Analyze a resume before preparing a resume report.")
		resume_pdf = st.session_state.get(f"jobs_report_pdf_resume_{user_id}")
		if resume_pdf:
			st.download_button("Download Resume Report PDF", resume_pdf, file_name="careerai_resume_report.pdf", mime="application/pdf", key=f"download_resume_pdf_{user_id}", on_click=record_activity, args=(user_id, "PDF downloaded", "Jobs & Reports", "Resume report PDF downloaded."))
		if st.button("Prepare Career Report PDF", key=f"prepare_career_pdf_{user_id}"):
			st.session_state[f"jobs_report_pdf_career_{user_id}"] = build_career_report_pdf(user, user_id)
			record_activity(user_id, "Career report generated", "Jobs & Reports", "Personalized career report PDF prepared.")
		career_pdf = st.session_state.get(f"jobs_report_pdf_career_{user_id}")
		if career_pdf:
			st.download_button("Download Career Report PDF", career_pdf, file_name="careerai_career_report.pdf", mime="application/pdf", key=f"download_career_pdf_{user_id}", on_click=record_activity, args=(user_id, "PDF downloaded", "Jobs & Reports", "Career report PDF downloaded."))

	# TAB BODY: Email Reports
	with email_tab:
		st.subheader("Email Reports")
		if not email_reports_configured():
			st.info("Email delivery is not configured. Set EMAIL_HOST, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_FROM as environment variables to enable sending.")
		report_choice = st.selectbox("Report to email", ["Progress Report", "Resume Report", "Career Report"], key=f"email_report_choice_{user_id}")
		recipient = st.text_input("Recipient email", value=str(user.get("email", "")), key=f"email_report_recipient_{user_id}")
		if st.button("Generate and send report", type="primary", key=f"send_jobs_report_{user_id}"):
			try:
				if report_choice == "Progress Report":
					pdf_bytes = build_progress_report_pdf(user, user_id)
				elif report_choice == "Resume Report":
					latest_analysis = get_analysis(user_id)
					if not latest_analysis:
						st.warning("Analyze a resume before emailing a resume report.")
						pdf_bytes = b""
					else:
						pdf_bytes = generate_resume_report(user, latest_analysis)
				else:
					pdf_bytes = build_career_report_pdf(user, user_id)
				if pdf_bytes:
					sent, message = send_report_email(user_id, recipient, report_choice, pdf_bytes)
					(st.success if sent else st.error)(message)
			except (OSError, ValueError, PermissionError) as error:
				st.error(f"Could not prepare the report: {error}")
		with connection() as database:
			email_rows = database.execute(
				"SELECT report_type, recipient, status, created_at FROM email_report_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
				(user_id,),
			).fetchall()
		if email_rows:
			st.markdown("**Recent email delivery history**")
			st.dataframe(pd.DataFrame([dict(row) for row in email_rows]), width="stretch", hide_index=True)


# TAB: Resources
# This page lists curated learning resources, bookmarks, and progress tracking.
def render_resources_page(user: dict[str, Any]) -> None:
	user_id = int(user["id"])
	st.header("Resources")
	st.caption("Curated learning materials, references, and portfolio-building assets for your target career path.")

	categories = list_resource_categories()
	resource_types = list_resource_types()
	category_choices = ["All categories"] + [item["name"] for item in categories]
	type_choices = ["All types"] + [item["type_name"] for item in resource_types]
	stats = get_resource_summary(user_id)
	metric_cols = st.columns(4)
	with metric_cols[0]:
		st.metric("Resources", str(stats["total"]), border=True)
	with metric_cols[1]:
		st.metric("Bookmarked", str(stats["bookmarked"]), border=True)
	with metric_cols[2]:
		st.metric("Completed", str(stats["completed"]), border=True)
	with metric_cols[3]:
		st.metric("In progress", str(stats["in_progress"]), border=True)

	with st.container(border=True):
		search_col, category_col, type_col = st.columns([2, 1, 1])
		with search_col:
			search_term = st.text_input(
				"Search resources",
				placeholder="Python, SQL, AI, interview, dataset...",
				key=f"resource_search_{user_id}",
			)
		with category_col:
			category_filter = st.selectbox(
				"Category",
				category_choices,
				index=0,
				key=f"resource_category_filter_{user_id}",
			)
		with type_col:
			type_filter = st.selectbox(
				"Type",
				type_choices,
				index=0,
				key=f"resource_type_filter_{user_id}",
			)
	resources = get_resources_for_user(
		user_id,
		category_name=category_filter,
		type_name=type_filter,
		search_term=search_term,
	)
	st.caption(f"Showing {len(resources)} resource{'s' if len(resources) != 1 else ''}")

	if not resources:
		st.info("No resources match your current filters. Try a broader keyword or change the category.", icon=":material/search:")
		return

	for start in range(0, len(resources), 2):
		card_columns = st.columns(2)
		for column, resource in zip(card_columns, resources[start:start + 2]):
			with column:
				with st.container(border=True):
					st.badge(resource["type_name"], icon=":material/menu_book:")
					st.markdown(f"### {resource['title']}")
					st.caption(
						f"{resource['category_name']} · {resource['difficulty']} · {resource['source']}"
					)
					st.write(resource["description"])
					st.caption(f"**Technology:** {resource['technology']} · **Topic:** {resource['topic']}")
					action_columns = st.columns(2)
					with action_columns[0]:
						if resource.get("url"):
							st.link_button(
								"Open resource",
								resource["url"],
								width="stretch",
								icon=":material/open_in_new:",
							)
					with action_columns[1]:
						bookmark_label = "Bookmarked" if resource["is_bookmarked"] else "Bookmark"
						bookmark_icon = ":material/bookmark:" if resource["is_bookmarked"] else ":material/bookmark_add:"
						if st.button(
							bookmark_label,
							key=f"resource_bookmark_{user_id}_{resource['id']}",
							width="stretch",
							icon=bookmark_icon,
						):
							toggle_resource_bookmark(user_id, resource["id"])
							st.rerun()

					with st.expander("Track learning progress"):
						with st.form(f"resource_progress_form_{user_id}_{resource['id']}"):
							status_options = ["Not Started", "In Progress", "Completed"]
							status_index = status_options.index(resource["status"]) if resource["status"] in status_options else 0
							status_value = st.selectbox(
								"Status",
								status_options,
								index=status_index,
								key=f"resource_status_{user_id}_{resource['id']}",
							)
							progress_value = st.slider(
								"Progress",
								0,
								100,
								int(resource["progress_percent"]),
								key=f"resource_progress_slider_{user_id}_{resource['id']}",
							)
							submitted = st.form_submit_button(
								"Save progress",
								width="stretch",
								icon=":material/save:",
							)
						if submitted:
							set_resource_progress(user_id, resource["id"], status_value, progress_value)
							st.success(f"Saved progress for {resource['title']}.")
							st.rerun()


# ---------------------------------------------------------------------------
# TAB 11: Admin Panel
# ---------------------------------------------------------------------------

def render_admin_users(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("User Management")
	search = st.text_input("Search users", placeholder="Name, email, or user ID", key="admin_user_search")
	role_filter = st.selectbox("Role filter", ["All roles", "user", "admin"], key="admin_user_role_filter")
	status_filter = st.selectbox("Account filter", ["All accounts", "Active", "Inactive"], key="admin_user_status_filter")
	rows = admin_user_rows(admin_id)
	filtered = [
		row for row in rows
		if (not search or search.casefold() in f"{row['id']} {row['full_name']} {row['email']}".casefold())
		and (role_filter == "All roles" or row["role"] == role_filter)
		and (status_filter == "All accounts" or bool(row["is_active"]) == (status_filter == "Active"))
	]
	if not filtered:
		st.info("No users match the selected filters.")
	else:
		page_size = 25
		page_count = max(1, (len(filtered) + page_size - 1) // page_size)
		page = int(st.number_input("User page", min_value=1, max_value=page_count, step=1, key="admin_user_page"))
		page_rows = filtered[(page - 1) * page_size:page * page_size]
		st.caption(f"Showing {(page - 1) * page_size + 1}–{min(page * page_size, len(filtered))} of {len(filtered)} users.")
		display_rows = [{
			"User ID": row["id"],
			"Name": row["full_name"],
			"Email": row["email"],
			"Role": row["role"].upper(),
			"Account": "Active" if row["is_active"] else "Inactive",
			"Registered": row["created_at"],
			"Last activity": row["last_activity"] or "No recorded activity",
			"Profile completion": f"{row['profile_completion']}%",
			"Resume score": row["resume_score"] if row["resume_score"] is not None else "No analysis",
			"Career readiness": f"{row['career_readiness']}%" if row["career_readiness"] is not None else "Insufficient data",
		} for row in page_rows]
		st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)
		user_choices = {
			f"{row['full_name']} · {row['email']} (ID {row['id']})": row
			for row in page_rows
		}
		selected_label = st.selectbox("Manage account", list(user_choices), key="admin_selected_user")
		selected = user_choices[selected_label]
		with st.expander("Profile summary and recent activity"):
			profile = get_user_profile(int(selected["id"]))
			st.write(f"**Career goal:** {profile['career_goal'] or 'Not provided'}")
			st.write(f"**Education:** {profile['education'] or 'Not provided'}")
			st.write(f"**Skills:** {', '.join(profile['skills']) if profile['skills'] else 'Not provided'}")
			activities = get_activity_history(int(selected["id"]))[:10]
			if activities:
				st.dataframe(pd.DataFrame([{
					"Activity": item["activity"],
					"Module": item["module"],
					"Status": item["status"],
					"Date": item["created_at"],
				} for item in activities]), width="stretch", hide_index=True)
			else:
				st.info("No recorded user activity.")
		with st.form(f"admin_user_edit_{selected['id']}"):
			role = st.selectbox("Role", ["user", "admin"], index=0 if selected["role"] == "user" else 1)
			active = st.checkbox("Account active", value=bool(selected["is_active"]))
			confirmed = st.checkbox("Confirm account change")
			submitted = st.form_submit_button("Save account changes", type="primary", disabled=not confirmed)
		if submitted:
			try:
				admin_update_user(admin_id, int(selected["id"]), role, active)
				st.success("Account updated.")
				st.rerun()
			except (ValueError, PermissionError) as error:
				st.error(str(error))


def render_admin_courses(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("Course Management")
	courses = admin_course_rows(admin_id)
	st.caption("Manage existing Learning Hub courses; inactive courses are hidden from learners.")
	if courses:
		st.dataframe(pd.DataFrame([{
			"Course": row["course_name"],
			"Category": row["category_name"],
			"Difficulty": row["difficulty"],
			"Topics": row["topic_count"],
			"Projects": row["project_count"],
			"Status": "Active" if row["active"] else "Inactive",
		} for row in courses]), width="stretch", hide_index=True)
	course_options = {"Create new course": None}
	course_options.update({f"{row['course_name']} · {row['category_name']} (ID {row['id']})": row for row in courses})
	selected_name = st.selectbox("Course details", list(course_options), key="admin_course_edit_select")
	selected = course_options[selected_name]
	with st.form("admin_course_form"):
		category_name = st.text_input("Category", value=selected["category_name"] if selected else "", max_chars=80)
		course_name = st.text_input("Course name", value=selected["course_name"] if selected else "", max_chars=120)
		description = st.text_area("Description", value=selected["description"] if selected else "", max_chars=4000)
		difficulty_options = ["Beginner", "Intermediate", "Advanced"]
		current_difficulty = selected["difficulty"] if selected else "Beginner"
		difficulty = st.selectbox("Difficulty", difficulty_options, index=difficulty_options.index(current_difficulty) if current_difficulty in difficulty_options else 0)
		duration = st.text_input("Duration", value=selected["duration"] if selected else "2 weeks", max_chars=80)
		tags = st.text_input("Tags (comma separated)", value=", ".join(json.loads(selected["tags_json"] or "[]")) if selected else "")
		save = st.form_submit_button("Save course", type="primary")
	if save:
		try:
			admin_save_course(
				admin_id, int(selected["id"]) if selected else None, category_name, course_name,
				description, difficulty, duration.strip() or "2 weeks",
				list(dict.fromkeys(item.strip() for item in tags.split(",") if item.strip())),
			)
			st.success("Course saved.")
			st.rerun()
		except (ValueError, sqlite3.IntegrityError, PermissionError) as error:
			st.error(f"Could not save course: {error}")
	if selected:
		left, right = st.columns([1, 2])
		with left:
			action = "Deactivate" if selected["active"] else "Activate"
			if st.button(f"{action} course", key=f"admin_course_active_{selected['id']}"):
				try:
					admin_set_course_active(admin_id, int(selected["id"]), not bool(selected["active"]))
					st.success(f"Course {action.lower()}d.")
					st.rerun()
				except (ValueError, PermissionError) as error:
					st.error(str(error))
		with right:
			st.caption("Existing learning progress is preserved when editing course content.")
		st.markdown("#### Topics, lessons, and projects")
		topics = []
		with connection() as database:
			topics = [dict(row) for row in database.execute(
				"SELECT id, topic_name, explanation, notes_json, examples_json, questions_json FROM learning_topics WHERE course_id = ? ORDER BY id",
				(int(selected["id"]),),
			).fetchall()]
		topic_options = {"Add topic": None}
		topic_options.update({f"{topic['topic_name']} (ID {topic['id']})": topic for topic in topics})
		topic_label = st.selectbox("Topic to edit", list(topic_options), key=f"admin_topic_select_{selected['id']}")
		topic = topic_options[topic_label]
		with st.form(f"admin_topic_form_{selected['id']}"):
			topic_title = st.text_input("Topic title", value=topic["topic_name"] if topic else "")
			topic_explanation = st.text_area("Topic explanation", value=topic["explanation"] if topic else "")
			topic_notes = st.text_area("Notes (one per line)", value="\n".join(json.loads(topic["notes_json"] or "[]")) if topic else "")
			topic_examples = st.text_area("Examples (one per line)", value="\n".join(json.loads(topic["examples_json"] or "[]")) if topic else "")
			topic_questions = st.text_area("Questions (one per line)", value="\n".join(json.loads(topic["questions_json"] or "[]")) if topic else "")
			save_topic = st.form_submit_button("Save topic")
		if save_topic:
			try:
				admin_save_topic(
					admin_id, int(selected["id"]), int(topic["id"]) if topic else None, topic_title,
					topic_explanation,
					[item.strip() for item in topic_notes.splitlines() if item.strip()],
					[item.strip() for item in topic_examples.splitlines() if item.strip()],
					[item.strip() for item in topic_questions.splitlines() if item.strip()],
				)
				st.success("Topic saved.")
				st.rerun()
			except (ValueError, sqlite3.IntegrityError, PermissionError) as error:
				st.error(f"Could not save topic: {error}")
		if topic:
			with connection() as database:
				lessons = [dict(row) for row in database.execute(
					"SELECT id, lesson_title, lesson_content FROM learning_lessons WHERE topic_id = ? ORDER BY id",
					(int(topic["id"]),),
				).fetchall()]
			lesson_options = {"Add lesson": None}
			lesson_options.update({f"{lesson['lesson_title']} (ID {lesson['id']})": lesson for lesson in lessons})
			lesson_label = st.selectbox("Lesson to edit", list(lesson_options), key=f"admin_lesson_select_{topic['id']}")
			lesson = lesson_options[lesson_label]
			with st.form(f"admin_lesson_form_{topic['id']}"):
				lesson_title = st.text_input("Lesson title", value=lesson["lesson_title"] if lesson else "")
				lesson_content = st.text_area("Lesson content", value=lesson["lesson_content"] if lesson else "")
				save_lesson = st.form_submit_button("Save lesson")
			if save_lesson:
				try:
					admin_save_lesson(admin_id, int(topic["id"]), int(lesson["id"]) if lesson else None, lesson_title, lesson_content)
					st.success("Lesson saved.")
					st.rerun()
				except (ValueError, sqlite3.IntegrityError, PermissionError) as error:
					st.error(f"Could not save lesson: {error}")
		with connection() as database:
			projects = [dict(row) for row in database.execute(
				"""SELECT id, project_title, difficulty, technologies_json, objective,
					required_skills_json, project_steps_json, expected_outcome
				   FROM learning_projects WHERE course_id = ? ORDER BY id""",
				(int(selected["id"]),),
			).fetchall()]
		project_options = {"Add project": None}
		project_options.update({f"{project['project_title']} (ID {project['id']})": project for project in projects})
		project_label = st.selectbox("Project to edit", list(project_options), key=f"admin_project_select_{selected['id']}")
		project = project_options[project_label]
		with st.form(f"admin_project_form_{selected['id']}"):
			project_title = st.text_input("Project title", value=project["project_title"] if project else "")
			project_difficulty_options = ["Beginner", "Intermediate", "Advanced"]
			project_difficulty_value = project["difficulty"] if project else "Beginner"
			project_difficulty = st.selectbox(
				"Project difficulty", project_difficulty_options,
				index=project_difficulty_options.index(project_difficulty_value) if project_difficulty_value in project_difficulty_options else 0,
			)
			project_technologies = st.text_input(
				"Technologies (comma separated)",
				value=", ".join(json.loads(project["technologies_json"] or "[]")) if project else "",
			)
			project_objective = st.text_area("Objective", value=project["objective"] if project else "")
			project_skills = st.text_input(
				"Required skills (comma separated)",
				value=", ".join(json.loads(project["required_skills_json"] or "[]")) if project else "",
			)
			project_steps = st.text_area(
				"Project steps (one per line)",
				value="\n".join(json.loads(project["project_steps_json"] or "[]")) if project else "",
			)
			project_outcome = st.text_area("Expected outcome", value=project["expected_outcome"] if project else "")
			save_project = st.form_submit_button("Save project")
		if save_project:
			try:
				admin_save_project(
					admin_id, int(selected["id"]), int(project["id"]) if project else None, project_title,
					project_difficulty,
					[item.strip() for item in project_technologies.split(",") if item.strip()],
					project_objective,
					[item.strip() for item in project_skills.split(",") if item.strip()],
					[item.strip() for item in project_steps.splitlines() if item.strip()],
					project_outcome,
				)
				st.success("Project saved.")
				st.rerun()
			except (ValueError, sqlite3.IntegrityError, PermissionError) as error:
				st.error(f"Could not save project: {error}")


def render_admin_resources(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("Resource Management")
	search = st.text_input("Search resources", placeholder="Title, topic, or technology", key="admin_resource_search")
	status = st.selectbox("Resource status", ["All", "Active", "Inactive"], key="admin_resource_status")
	resources = admin_resource_rows(admin_id)
	filtered = [
		row for row in resources
		if (not search or search.casefold() in f"{row['title']} {row['topic']} {row['technology']}".casefold())
		and (status == "All" or bool(row["active"]) == (status == "Active"))
	]
	if filtered:
		page_size = 25
		page_count = max(1, (len(filtered) + page_size - 1) // page_size)
		page = int(st.number_input("Resource page", min_value=1, max_value=page_count, step=1, key="admin_resource_page"))
		page_rows = filtered[(page - 1) * page_size:page * page_size]
		st.caption(f"Showing {(page - 1) * page_size + 1}–{min(page * page_size, len(filtered))} of {len(filtered)} resources.")
		st.dataframe(pd.DataFrame([{
			"ID": row["id"], "Title": row["title"], "Category": row["category_name"],
			"Type": row["type_name"], "Topic": row["topic"], "Difficulty": row["difficulty"],
			"Status": "Active" if row["active"] else "Inactive",
			"Created": row["created_at"], "Updated": row["updated_at"],
		} for row in page_rows]), width="stretch", hide_index=True)
	resource_options = {"Create resource": None}
	resource_options.update({f"{row['title']} · {row['category_name']} (ID {row['id']})": row for row in resources})
	selected_label = st.selectbox("Resource details", list(resource_options), key="admin_resource_edit_select")
	selected = resource_options[selected_label]
	with st.form("admin_resource_form"):
		title = st.text_input("Title", value=selected["title"] if selected else "", max_chars=160)
		description = st.text_area("Description", value=selected["description"] if selected else "", max_chars=4000)
		category = st.text_input("Category", value=selected["category_name"] if selected else "")
		resource_type = st.text_input("Resource type", value=selected["type_name"] if selected else "")
		technology = st.text_input("Technology", value=selected["technology"] if selected else "")
		topic = st.text_input("Topic", value=selected["topic"] if selected else "")
		difficulties = ["Beginner", "Intermediate", "Advanced"]
		current_difficulty = selected["difficulty"] if selected else "Beginner"
		difficulty = st.selectbox("Difficulty", difficulties, index=difficulties.index(current_difficulty) if current_difficulty in difficulties else 0)
		url = st.text_input("HTTPS URL (optional)", value=selected["url"] or "" if selected else "")
		st.caption("Only link to content you own or are authorized to share. Uploads are not supported in this admin form.")
		save = st.form_submit_button("Save resource", type="primary")
	if save:
		try:
			admin_save_resource(
				admin_id, int(selected["id"]) if selected else None,
				title, description, category, resource_type, technology, topic, difficulty, url,
			)
			st.success("Resource saved.")
			st.rerun()
		except (ValueError, sqlite3.IntegrityError, PermissionError) as error:
			st.error(f"Could not save resource: {error}")
	if selected:
		left, right = st.columns(2)
		with left:
			action = "Deactivate" if selected["active"] else "Activate"
			if st.button(f"{action} resource", key=f"admin_resource_active_{selected['id']}"):
				try:
					admin_set_resource_active(admin_id, int(selected["id"]), not bool(selected["active"]))
					st.success(f"Resource {action.lower()}d.")
					st.rerun()
				except (ValueError, PermissionError) as error:
					st.error(str(error))
		with right:
			confirm_delete = st.checkbox("Confirm permanent deletion", key=f"confirm_delete_resource_{selected['id']}")
			if st.button("Delete unused resource", disabled=not confirm_delete, key=f"admin_resource_delete_{selected['id']}"):
				try:
					admin_delete_resource(admin_id, int(selected["id"]))
					st.success("Resource deleted.")
					st.rerun()
				except (ValueError, PermissionError) as error:
					st.error(str(error))


def render_admin_analytics(admin: dict[str, Any]) -> None:
	st.subheader("Platform Analytics")
	stats = admin_analytics(int(admin["id"]))
	metrics = [
		("Total users", stats["users"]),
		("Active users", stats["active_users"]),
		("New users · 30 days", stats["new_users"]),
		("Resume analyses", stats["resume_analyses"]),
		("Average ATS score", f"{stats['average_ats']:.1f}" if stats["average_ats"] is not None else "No data"),
		("Average career readiness", f"{stats['average_readiness']}%" if stats["average_readiness"] is not None else "No data"),
		("Active courses", stats["courses"]),
		("Active resources", stats["resources"]),
		("Course completions", stats["course_completions"]),
		("Quiz / MCQ attempts", stats["quiz_attempts"]),
		("Coding attempts", stats["coding_attempts"]),
		("Mock interview attempts", stats["mock_interviews"]),
		("Career recommendations", stats["career_recommendations"]),
		("Personal reports generated", stats["reports_generated"]),
		("User feedback", stats["feedback_count"]),
		("Admin reports generated", stats["admin_reports_generated"]),
	]
	for offset in range(0, len(metrics), 4):
		columns = st.columns(4)
		for column, (label, value) in zip(columns, metrics[offset:offset + 4]):
			column.metric(label, value)
	st.caption("Job/internship recommendations are not stored as separate records in this app, so no count is shown.")
	charts = [
		("User growth", stats["user_growth"], "day", "users"),
		("Successful active-user logins", stats["active_growth"], "day", "active_users"),
		("Resume analyses and ATS scores", stats["resume_activity"], "day", "analyses"),
		("Learning activity", stats["learning_activity"], "day", "updates"),
	]
	for offset in range(0, len(charts), 2):
		columns = st.columns(2)
		for column, (title, values, x_field, y_field) in zip(columns, charts[offset:offset + 2]):
			with column:
				with st.container(border=True):
					st.markdown(f"**{title}**")
					if values:
						frame = pd.DataFrame(values)
						figure = px.line(frame, x=x_field, y=y_field, markers=True, title=title)
						column.plotly_chart(figure, width="stretch")
						if title == "Resume analyses and ATS scores" and frame["average_ats"].notna().any():
							column.caption(f"Overall average ATS score: {stats['average_ats']:.1f}/100")
						elif title == "Learning activity" and frame["average_progress"].notna().any():
							column.caption("Daily average saved learning progress is included in the source data.")
					else:
						column.info("No data available yet.")


def render_admin_feedback(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("Feedback Management")
	with st.expander("Record feedback received from a user"):
		with connection() as database:
			users = [dict(row) for row in database.execute(
				"SELECT id, full_name, email FROM users ORDER BY full_name"
			).fetchall()]
		if not users:
			st.info("There are no user accounts to associate with feedback.")
		else:
			user_options = {f"{row['full_name']} · {row['email']} (ID {row['id']})": int(row["id"]) for row in users}
			with st.form("admin_feedback_create_form"):
				feedback_user = st.selectbox("User", list(user_options))
				feedback_type = st.text_input("Feedback type", max_chars=80, placeholder="Suggestion, issue, or compliment")
				message = st.text_area("Feedback message", max_chars=4000)
				rating_choice = st.selectbox("Rating", ["Not provided", "1", "2", "3", "4", "5"])
				create = st.form_submit_button("Record feedback")
			if create:
				try:
					admin_add_feedback(
						admin_id, user_options[feedback_user], feedback_type, message,
						int(rating_choice) if rating_choice != "Not provided" else None,
					)
					st.success("Feedback recorded.")
					st.rerun()
				except (ValueError, PermissionError) as error:
					st.error(str(error))
	search = st.text_input("Search feedback", placeholder="Type, user, or message", key="admin_feedback_search")
	status_filter = st.selectbox("Feedback status", ["All", "New", "Reviewed", "In Progress", "Resolved"], key="admin_feedback_status")
	rows = admin_feedback_rows(admin_id)
	filtered = [
		row for row in rows
		if (not search or search.casefold() in f"{row['feedback_type']} {row['user_name']} {row['message']}".casefold())
		and (status_filter == "All" or row["status"] == status_filter)
	]
	if not filtered:
		st.info("No feedback records match these filters. Feedback will appear here when user-submission records exist.")
		return
	page_size = 25
	page_count = max(1, (len(filtered) + page_size - 1) // page_size)
	page = int(st.number_input("Feedback page", min_value=1, max_value=page_count, step=1, key="admin_feedback_page"))
	page_rows = filtered[(page - 1) * page_size:page * page_size]
	st.caption(f"Showing {(page - 1) * page_size + 1}–{min(page * page_size, len(filtered))} of {len(filtered)} feedback records.")
	st.dataframe(pd.DataFrame([{
		"ID": row["id"], "User": row["user_name"], "Type": row["feedback_type"],
		"Message": row["message"], "Rating": row["rating"] or "Not provided",
		"Date": row["created_at"], "Status": row["status"],
	} for row in page_rows]), width="stretch", hide_index=True)
	options = {f"#{row['id']} · {row['feedback_type']} · {row['user_name']}": row for row in page_rows}
	selected_label = st.selectbox("Review feedback", list(options), key="admin_feedback_selected")
	selected = options[selected_label]
	st.write(selected["message"])
	with st.form(f"admin_feedback_form_{selected['id']}"):
		status_options = ["New", "Reviewed", "In Progress", "Resolved"]
		new_status = st.selectbox("Status", status_options, index=status_options.index(selected["status"]))
		note = st.text_area("Internal admin note", value=selected["admin_note"], max_chars=2000)
		save = st.form_submit_button("Save feedback review")
	if save:
		try:
			admin_update_feedback(admin_id, int(selected["id"]), new_status, note)
			st.success("Feedback review saved.")
			st.rerun()
		except (ValueError, PermissionError) as error:
			st.error(str(error))


def render_admin_models(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("AI Model Management")
	st.caption("Service descriptions reflect the code in use. This page does not train models or display secret values.")
	models = admin_model_rows(admin_id)
	if not models:
		st.info("No AI services are registered.")
		return
	st.dataframe(pd.DataFrame([{
		"Service": row["service_name"], "Purpose": row["purpose"],
		"Technology": row["technology"], "Version": row["version"],
		"Type": row["model_type"],
		"Status": ("External calls permitted" if row["enabled"] else "External calls disabled")
			if row["service_key"] == "ai_assistant" else "Active",
		"Last updated": row["updated_at"],
	} for row in models]), width="stretch", hide_index=True)
	assistant = next((row for row in models if row["service_key"] == "ai_assistant"), None)
	if assistant:
		external_provider = get_ai_provider()
		st.caption(f"Configured provider: {external_provider}. API keys and secret values are not displayed.")
		enabled = st.toggle(
			"Allow external AI provider calls",
			value=bool(assistant["enabled"]),
			help="When disabled, the existing local assistant response continues to work.",
			key="admin_external_ai_enabled",
		)
		if enabled != bool(assistant["enabled"]):
			try:
				admin_set_external_ai_enabled(admin_id, enabled)
				st.success("External AI provider setting updated.")
				st.rerun()
			except PermissionError as error:
				st.error(str(error))
	options = {f"{row['service_name']} (ID {row['id']})": row for row in models}
	selected_label = st.selectbox("Edit service metadata", list(options), key="admin_model_edit")
	selected = options[selected_label]
	with st.form(f"admin_model_form_{selected['id']}"):
		version = st.text_input("Version / configured model", value=selected["version"], max_chars=100)
		technology = st.text_input("Technology", value=selected["technology"], max_chars=200)
		description = st.text_area("Description", value=selected["description"], max_chars=1000)
		save = st.form_submit_button("Save metadata")
	if save:
		try:
			admin_update_model(admin_id, int(selected["id"]), version, technology, description)
			st.success("AI service metadata updated.")
			st.rerun()
		except (ValueError, PermissionError) as error:
			st.error(str(error))


def render_admin_reports(admin: dict[str, Any]) -> None:
	admin_id = int(admin["id"])
	st.subheader("Platform Reports")
	report_types = [
		"User Report", "Learning Report", "Resume Analysis Report",
		"Career Recommendation Report", "Practice & Interview Report",
		"Job/Internship Recommendation Report", "Feedback Report", "Platform Activity Report",
	]
	report_type = st.selectbox("Report type", report_types, key="admin_report_type")
	default_start = date.today() - timedelta(days=30)
	date_range = st.date_input("Date range", (default_start, date.today()), key="admin_report_dates")
	if isinstance(date_range, tuple) and len(date_range) == 2:
		start_date, end_date = date_range
	else:
		start_date = end_date = date_range if isinstance(date_range, date) else date.today()
	if start_date > end_date:
		st.error("The report start date must be on or before the end date.")
		return
	user_options: dict[str, int | None] = {"All users": None}
	with connection() as database:
		users = database.execute("SELECT id, full_name, email FROM users ORDER BY full_name").fetchall()
	for row in users:
		user_options[f"{row['full_name']} · {row['email']} (ID {row['id']})"] = int(row["id"])
	selected_user = st.selectbox("User filter", list(user_options), key="admin_report_user")
	module_filter = "All modules"
	if report_type == "Platform Activity Report":
		module_filter = st.selectbox(
			"Module filter", ["All modules", "Dashboard", "My Profile", "Resume", "Career Guidance", "Learning Hub", "Practice & Interview", "Jobs & Reports", "Resources", "AI Assistant"],
			key="admin_report_module",
		)
	try:
		report_frame = admin_report_data(
			admin_id, report_type, start_date, end_date, user_options[selected_user],
		)
	except (ValueError, PermissionError, sqlite3.Error) as error:
		st.error(f"Could not load report data: {error}")
		return
	if module_filter != "All modules" and "module" in report_frame.columns:
		report_frame = report_frame[report_frame["module"] == module_filter]
	report_filters = (
		report_type, start_date.isoformat(), end_date.isoformat(),
		user_options[selected_user], module_filter,
	)
	st.caption(f"{len(report_frame)} record(s) match the selected report filters. Showing at most 500 records.")
	if report_frame.empty:
		st.info("No data available for this report and date range.")
	else:
		st.dataframe(report_frame, width="stretch", hide_index=True)
	if st.button("Generate PDF report", type="primary", key="admin_generate_report"):
		try:
			sections = [
				("Summary", [
					f"Report type: {report_type}",
					f"Date range: {start_date.isoformat()} through {end_date.isoformat()}",
					f"Matching records: {len(report_frame)}",
					f"User filter: {selected_user}",
					f"Module filter: {module_filter}",
				]),
				("Report records", [
					" · ".join(f"{column}: {str(row[column])[:180]}" for column in report_frame.columns)
					for _, row in report_frame.head(100).iterrows()
				] or ["No records matched the selected filters."]),
			]
			pdf = build_jobs_reports_pdf(admin, report_type, sections)
			admin_log_report(admin_id, report_type, start_date, end_date)
			st.session_state["admin_generated_report"] = {
				"bytes": pdf,
				"filename": f"careerai_{re.sub(r'[^a-z0-9]+', '_', report_type.casefold()).strip('_')}_{date.today().isoformat()}.pdf",
				"filters": report_filters,
			}
			st.success("Report generated and recorded in the admin activity log.")
		except (ValueError, PermissionError, sqlite3.Error) as error:
			st.error(f"Could not generate report: {error}")
	generated = st.session_state.get("admin_generated_report")
	if generated and generated.get("filters") == report_filters:
		st.download_button(
			"Download PDF",
			data=generated["bytes"],
			file_name=generated["filename"],
			mime="application/pdf",
			key="admin_download_report",
		)
	st.markdown("#### Recent admin actions")
	with connection() as database:
		log_rows = [dict(row) for row in database.execute(
			"""SELECT l.action, l.module, l.target_record, l.created_at, u.full_name AS admin
			   FROM admin_activity_logs l JOIN users u ON u.id = l.admin_user_id
			   ORDER BY l.created_at DESC LIMIT 50"""
		).fetchall()]
	if log_rows:
		st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)
	else:
		st.info("No admin actions have been logged yet.")


def render_admin_panel(user: dict[str, Any]) -> None:
	try:
		admin = require_admin(int(user["id"]))
	except (KeyError, TypeError, ValueError, PermissionError):
		st.error("Access Denied — Administrator privileges required.")
		st.caption("Admin Panel sab users ko dikhai deta hai, lekin iska access sirf authorized admin account ko hai.")
		return
	st.title("Admin Panel")
	st.caption(f"Administration · {admin['full_name']}")
	section = st.segmented_control(
		"Admin sections",
		["Analytics", "User Management", "Course Management", "Resource Management", "Feedback", "AI Model Management", "Reports"],
		default="Analytics",
		key="admin_panel_section",
	)
	if section == "Analytics":
		render_admin_analytics(admin)
	elif section == "User Management":
		render_admin_users(admin)
	elif section == "Course Management":
		render_admin_courses(admin)
	elif section == "Resource Management":
		render_admin_resources(admin)
	elif section == "Feedback":
		render_admin_feedback(admin)
	elif section == "AI Model Management":
		render_admin_models(admin)
	elif section == "Reports":
		render_admin_reports(admin)


ABOUT_MODULES = [
	("Login / Authentication", "Create an account, verify email, sign in, and securely manage access.", "Account access"),
	("Dashboard", "Review your profile completion, saved resume scores, career readiness, and recent activity.", "Personal overview"),
	("My Profile", "Maintain your education, experience, skills, interests, and career goal.", "Career profile"),
	("Resume", "Extract skills and experience, compare them with a pasted job description, identify skill gaps, and show role-specific suggestions alongside the rule-based ATS score.", "Resume insights"),
	("Career Guidance", "Explore saved career recommendations, skill gaps, career roadmaps, and learning paths.", "Career planning"),
	("Learning Hub", "Browse courses and topics and save your learning progress.", "Structured learning"),
	("Resources", "Find categorized learning resources, bookmark them, and track progress.", "Study materials"),
	("AI Assistant", "Ask career and learning questions, get study guidance, and use configured Groq/OpenAI providers or local responses.", "Career assistant"),
	("Practice & Interview", "Try quizzes, coding exercises, interview questions, and keyword-based mock interviews.", "Skill practice"),
	("Jobs & Reports", "Review curated sample role matches, saved activity, and generate personal PDF reports.", "Progress and reports"),
	("Admin Panel", "Manage users, courses, resources, feedback, AI service settings, analytics, and platform reports.", "Administration"),
	("About", "Learn about CareerAI, its technologies, support guidance, developer, and version.", "Project information"),
]

ABOUT_FAQ = [
	(
		"What is CareerAI?",
		"CareerAI is a career-development platform that brings together resume analysis, profile management, "
		"career guidance, structured learning, practice, and progress reports.",
	),
	(
		"Who can use CareerAI?",
		"Students and job seekers can create a user account to organize career goals, review their resume, "
		"and explore learning and practice tools.",
	),
	(
		"How does CareerAI analyze resumes?",
		"CareerAI extracts text from uploaded PDFs, checks for known sections and skills, and compares the "
		"results with its configured career skill maps. Scoring and suggestions are produced by application "
		"rules; the current project does not train a resume-scoring model.",
	),
	(
		"What is an ATS score?",
		"It is CareerAI's 0–100 rule-based estimate of resume skills, keywords, structure, education, "
		"experience, projects, and completeness. It is guidance, not a guarantee of how an employer's ATS "
		"will rank an application.",
	),
	(
		"How are career recommendations generated?",
		"CareerAI compares saved profile and resume skills with configured career-role requirements and "
		"stores the resulting role matches. It does not predict hiring outcomes.",
	),
	(
		"What is skill gap analysis?",
		"It compares available profile/resume skills with the requirements mapped to a career direction "
		"and lists matching and missing skills to help plan what to learn next.",
	),
	(
		"How does the Learning Hub work?",
		"It offers stored course, topic, lesson, and project content. Learner progress is saved against "
		"their account and can be reviewed in the Learning Hub and dashboard.",
	),
	(
		"What can the AI Assistant do?",
		"It provides career and learning responses, topic explanations, study-planning help, and related "
		"guidance. It uses local responses when no external provider is configured, or when external "
		"provider calls are disabled by an administrator.",
	),
	(
		"What is the AI Mock Interview?",
		"It is a practice feature that estimates answer coverage using configured keyword matching. "
		"It is not a trained interview-evaluation model and its feedback should be treated as practice guidance.",
	),
	(
		"Can CareerAI recommend jobs?",
		"Jobs & Reports shows matches against curated example job and internship role profiles. These are "
		"not live vacancies and are not fetched from an external job board.",
	),
	(
		"Can I download reports?",
		"Yes. Available personal resume, progress, and career reports can be generated as PDFs from "
		"Jobs & Reports. Admins can also generate filtered platform reports.",
	),
	(
		"Is my personal data secure?",
		"Accounts use hashed passwords, and personal records are associated with their account. Do not "
		"upload information you are not comfortable storing in this project deployment; administrators "
		"can access limited platform records for their duties.",
	),
	(
		"Is CareerAI using Machine Learning?",
		"The current resume scores, skill extraction, career matches, and mock interview feedback primarily "
		"use deterministic rules and text patterns. The project does not claim these components are trained "
		"ML models. The AI Assistant can optionally connect to an externally configured LLM provider.",
	),
	(
		"Can CareerAI be expanded in the future?",
		"Yes. The project can be extended with features such as a live job data integration, separately "
		"deployed API, model evaluation, or cloud hosting. Those are future work unless explicitly enabled "
		"and implemented in a deployment.",
	),
	(
		"What technologies are used to build CareerAI?",
		"The current app uses Python, Streamlit, SQLite, Pandas, PyMuPDF, Plotly, ReportLab, and Python "
		"standard libraries. Groq and OpenAI-compatible APIs are optional integrations. Other planned "
		"technologies are labelled separately below.",
	),
]


def configured_about_url(value: str) -> str | None:
	if not value:
		return None
	try:
		parsed = urlsplit(value)
	except ValueError:
		return None
	if parsed.scheme.casefold() != "https" or not parsed.hostname or parsed.username or parsed.password:
		return None
	return value


def render_about_page() -> None:
	st.title(f"About {CAREERAI_CONFIG['name']}")
	st.caption("Project overview, platform guide, developer details, and version information.")

	with st.container(border=True):
		st.subheader("About CareerAI")
		st.write(CAREERAI_CONFIG["description"])
		st.write(
			"CareerAI is intended to help students and job seekers understand how their saved profile "
			"and resume relate to configured career paths. It brings resume review, skill-gap guidance, "
			"learning content, practice, and reports together in one application."
		)
		st.write(
			"Text processing identifies resume sections and catalogued skills. Career matches and ATS "
			"scores use application-defined rules; the AI Assistant can use a configured external LLM "
			"or provide local responses. These tools support career preparation and do not guarantee "
			"employment or replace professional advice."
		)

	st.subheader("Features")
	for row_start in range(0, len(ABOUT_MODULES), 3):
		columns = st.columns(3)
		for column, (name, description, purpose) in zip(columns, ABOUT_MODULES[row_start:row_start + 3]):
			with column:
				with st.container(border=True):
					st.markdown(f"**:material/auto_awesome: {name}**")
					st.write(description)
					st.caption(f"Purpose · {purpose}")

	st.subheader("Technologies used")
	current_technologies = [
		("Python", "Application logic and existing resume/career rules."),
		("Streamlit", "User interface and navigation."),
		("SQLite", "Application data storage."),
		("Pandas", "Tabular data and report preparation."),
		("PyMuPDF", "PDF resume text extraction."),
		("Plotly", "Admin analytics charts."),
		("ReportLab", "PDF report generation."),
		("bcrypt", "Password hashing for account credentials."),
		("Python standard library", "Database, validation, JSON, security helpers, and optional email."),
	]
	optional_technologies = [
		("Groq API", "Optional AI Assistant provider when credentials are configured."),
		("OpenAI API", "Optional AI Assistant provider when credentials are configured."),
	]
	future_technologies = [
		"FastAPI API service", "MySQL", "Docker", "AWS", "OpenCV",
		"NumPy", "Scikit-learn model training", "NLTK", "Git / GitHub integrations",
	]
	st.markdown("**Currently used by the application**")
	for row_start in range(0, len(current_technologies), 4):
		columns = st.columns(4)
		for column, (name, detail) in zip(columns, current_technologies[row_start:row_start + 4]):
			with column:
				with st.container(border=True):
					st.badge(name, color="green")
					st.caption(detail)
	st.markdown("**Optional integrations**")
	integration_columns = st.columns(2)
	for column, (name, detail) in zip(integration_columns, optional_technologies):
		with column:
			with st.container(border=True):
				st.badge(name, color="blue")
				st.caption(detail)
	st.markdown("**Future / scalability technologies (not currently integrated)**")
	with st.container(border=True):
		st.write(" · ".join(f"`{name}`" for name in future_technologies))

	st.subheader("Contact")
	with st.container(border=True):
		st.markdown(f"**Developer:** {CAREERAI_CONFIG['developer_name']}")
		st.write(f"**Department:** {CAREERAI_CONFIG['department']}")
		st.write(f"**Enrollment number:** {CAREERAI_CONFIG['enrollment_number']}")
		contact_available = False
		contact_email = CAREERAI_CONFIG["contact_email"]
		if contact_email and validate_email(contact_email):
			st.markdown(f"**Email:** [{contact_email}](mailto:{contact_email})")
			contact_available = True
		contact_phone = CAREERAI_CONFIG["contact_phone"]
		if contact_phone:
			st.write(f"**Phone:** {contact_phone}")
			contact_available = True
		location = CAREERAI_CONFIG["location"]
		if location:
			st.write(f"**Location:** {location}")
			contact_available = True
		github_url = configured_about_url(CAREERAI_CONFIG["github_url"])
		linkedin_url = configured_about_url(CAREERAI_CONFIG["linkedin_url"])
		contact_links = []
		if github_url:
			contact_links.append(("GitHub", github_url))
		if linkedin_url:
			contact_links.append(("LinkedIn", linkedin_url))
		if contact_links:
			link_columns = st.columns(min(2, len(contact_links)))
			for column, (label, url) in zip(link_columns, contact_links):
				column.link_button(label, url, width="stretch")
			contact_available = True
		if not contact_available:
			st.info("Contact information will be available soon.")
	st.caption("To configure contact channels, set CAREERAI_CONTACT_EMAIL, CAREERAI_CONTACT_PHONE, CAREERAI_LOCATION, CAREERAI_GITHUB_URL, and CAREERAI_LINKEDIN_URL.")

	st.subheader("Help & Support")
	st.caption(CAREERAI_CONFIG["support_information"])
	help_topics = [
		(
			"Account and profile",
			"Create an account from the sign-up screen, verify your email using the provided verification flow, "
			"then log in. Use My Profile to save your education, experience, skills, interests, and career goal.",
		),
		(
			"Resume and ATS score",
			"Open Resume, upload a supported text-readable PDF, and start analysis. The ATS score summarizes "
			"the configured score dimensions; read the breakdown and suggestions rather than treating it as "
			"an employer's official ATS result.",
		),
		(
			"Career guidance and learning",
			"Add a career goal and skills in My Profile, then review Career Guidance for stored matches and "
			"skill gaps. Choose a course in Learning Hub and mark topics as complete to save learning progress.",
		),
		(
			"AI Assistant and practice",
			"Use AI Assistant for career and learning questions. Try quizzes, coding practice, interview "
			"questions, and mock interviews in Practice & Interview. Mock interview feedback is keyword-based.",
		),
		(
			"Jobs and reports",
			"Review example role matches in Jobs & Reports; they are curated examples, not live postings. "
			"Use the report controls to prepare available personal PDF reports.",
		),
		(
			"Troubleshooting",
			"If resume upload fails, confirm it is a supported PDF within the upload size limit and try a "
			"text-readable file. If the AI provider is unavailable, check its deployment configuration or "
			"use the local assistant. If PDF generation fails, retry after confirming the relevant saved "
			"analysis exists. If learning content is missing, refresh the app and contact the project "
			"maintainer. For login/session problems, confirm the account is active and sign in again.",
		),
	]
	for title, answer in help_topics:
		with st.expander(title):
			st.write(answer)

	st.subheader("Frequently asked questions")
	for question, answer in ABOUT_FAQ:
		with st.expander(question):
			st.write(answer)

	st.subheader("Version information")
	with st.container(border=True):
		version_columns = st.columns(3)
		version_columns[0].metric("Application", CAREERAI_CONFIG["name"])
		version_columns[1].metric("Version", CAREERAI_CONFIG["version"])
		version_columns[2].metric("Release status", CAREERAI_CONFIG["release_status"])
		st.write(f"**Project type:** {CAREERAI_CONFIG['project_type']}")
		st.write("**Technology:** Python and Streamlit")
		st.write(f"**Last updated:** {CAREERAI_CONFIG['last_updated']}")


# TAB: App Shell
# Main authenticated navigation shell for all user pages.
def authenticated_screen() -> None:
	user = st.session_state.auth_user
	with st.sidebar:
		st.markdown("<div class='brand-mark'>CareerAI</div>", unsafe_allow_html=True)
		st.caption("CAREER WORKSPACE")
		pages = [("Dashboard", ":material/dashboard:"), ("My Profile", ":material/person:"), ("Resume", ":material/description:"), ("Career Guidance", ":material/psychology:"), ("Learning Hub", ":material/school:"), ("Resources", ":material/library_books:"), ("Practice & Interview", ":material/quiz:"), ("AI Assistant", ":material/auto_awesome:"), ("Jobs & Reports", ":material/workspace_premium:")]
		pages.append(("Admin Panel", ":material/admin_panel_settings:"))
		pages.append(("About", ":material/info:"))
		st.session_state.setdefault("main_navigation", "Dashboard")
		for page_name, page_icon in pages:
			if st.button(
				page_name,
				icon=page_icon,
				type="primary" if st.session_state.main_navigation == page_name else "secondary",
				width="stretch",
				key=f"main_navigation_{page_name.lower().replace(' ', '_')}",
			):
				st.session_state.main_navigation = page_name
				st.rerun()
		st.divider()
		st.caption(user["full_name"])
		st.caption(user["email"])
		if st.button("Log out", icon=":material/logout:", width="stretch", key="logout_button"):
			st.session_state.auth_user = None
			st.session_state.auth_view = "login"
			st.session_state.pop("resume_report_ref", None)
			st.session_state.pop("resume_selected_analysis", None)
			for key in list(st.session_state):
				if key.startswith("resume_jd_match_"):
					st.session_state.pop(key, None)
			st.session_state.pop("main_navigation", None)
			st.rerun()
	selected = st.session_state.main_navigation
	if selected == "Dashboard":
		render_dashboard_page(user)
	elif selected == "My Profile":
		render_profile_page(user)
	elif selected == "Career Guidance":
		render_career_guidance_page(user)
	elif selected == "Learning Hub":
		render_learning_hub_page(user)
	elif selected == "Practice & Interview":
		render_practice_interview_page(user)
	elif selected == "Jobs & Reports":
		render_jobs_reports_page(user)
	elif selected == "Resources":
		render_resources_page(user)
	elif selected == "AI Assistant":
		render_ai_assistant_page(user)
	elif selected == "Admin Panel":
		render_admin_panel(user)
	elif selected == "About":
		render_about_page()
	else:
		render_resume_page(user)


# APP ENTRY POINT
# Initializes the app and routes between auth and authenticated UI.
def render_app() -> None:
	initialize_session()
	inject_styles()
	if st.session_state.auth_user:
		authenticated_screen()
	else:
		with st.sidebar:
			st.markdown("<div class='brand-mark'>CareerAI</div>", unsafe_allow_html=True)
			st.caption("CAREER WORKSPACE")
			auth_choices = {"login": "Log in", "signup": "Create account", "forgot": "Forgot password"}
			for view, label in auth_choices.items():
				if st.button(
					label,
					icon={"login": ":material/login:", "signup": ":material/person_add:", "forgot": ":material/lock_reset:"}[view],
					type="primary" if st.session_state.auth_view == view else "secondary",
					width="stretch",
					key=f"auth_navigation_{view}",
				):
					st.session_state.auth_view = view
					st.rerun()
		st.title("Your next chapter starts here")
		st.markdown('<p class="auth-subtitle">Intelligent tools for the career you are building.</p>', unsafe_allow_html=True)
		if st.session_state.auth_view == "login":
			login_screen()
		elif st.session_state.auth_view == "signup":
			signup_screen()
		else:
			forgot_screen()
		verification_screen()


if os.getenv("CAREERAI_ENABLE_API", "false").lower() != "true":
	initialize_database()
	st.set_page_config(page_title="CareerAI | Resume analysis", page_icon=":material/workspace_premium:", layout="wide")
	render_app()
 