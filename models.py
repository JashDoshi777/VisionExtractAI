# models.py - UPDATED

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
from sqlalchemy import Text
import json

db = SQLAlchemy()

# Business Card Model
class BusinessCard(db.Model):
    __tablename__ = 'business_cards'
    
    id = db.Column(db.Integer, primary_key=True)
    json_id = db.Column(db.String(64), unique=True, nullable=False)  
    owner_name = db.Column(db.String(200), nullable=False)
    company_name = db.Column(db.String(200))
    email = db.Column(db.String(120))
    phone_number = db.Column(db.String(50))
    address = db.Column(db.Text)
    source_document = db.Column(db.String(500))
    user_hash = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.json_id,
            'Owner Name': self.owner_name,
            'Company Name': self.company_name,
            'Email': self.email,
            'Number': self.phone_number,
            'Address': self.address,
            'source_document': self.source_document,
            'image_filename': f"{self.json_id}.png",
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Brochure Model
class Brochure(db.Model):
    __tablename__ = 'brochures'
    
    id = db.Column(db.Integer, primary_key=True)
    json_id = db.Column(db.String(64), unique=True, nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    raw_text = db.Column(db.Text)
    source_document = db.Column(db.String(500))
    user_hash = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    contacts = db.relationship('Contact', backref='brochure', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.json_id,
            'company_name': self.company_name,
            'raw_text': self.raw_text,
            'contacts': [contact.to_dict() for contact in self.contacts],
            'image_filename': f"{self.json_id}.pdf",
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Contact Model (for Brochures)
class Contact(db.Model):
    __tablename__ = 'brochure_contacts'
    
    id = db.Column(db.Integer, primary_key=True)
    json_id = db.Column(db.String(64), unique=True, nullable=False)
    brochure_id = db.Column(db.Integer, db.ForeignKey('brochures.id'), nullable=False)
    owner_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120))
    phone_number = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.json_id,
            'Owner Name': self.owner_name,
            'Email': self.email,
            'Number': self.phone_number,
            'brochure_id': self.brochure_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# User Model for CRM - ENHANCED
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120))
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    department = db.Column(db.String(100))
    role = db.Column(db.String(20), default='employee')
    status = db.Column(db.String(20), default='active')
    dark_mode = db.Column(db.Boolean, default=False)  # NEW: Dark mode preference
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    contacts_assigned = db.relationship('CRMContact', foreign_keys='CRMContact.assigned_to', backref='assigned_user', lazy=True)
    deals_assigned = db.relationship('Deal', foreign_keys='Deal.assigned_to', backref='assigned_user', lazy=True)
    tasks_assigned = db.relationship('Task', foreign_keys='Task.assigned_to', backref='assigned_user', lazy=True)
    activities = db.relationship('Activity', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    # In User model
    task_assignments_created = db.relationship(
    'TaskAssignment',
    foreign_keys='TaskAssignment.assigned_by',
    backref='assigner',
    lazy=True
    )

    task_assignments_received = db.relationship(
    'TaskAssignment',
    foreign_keys='TaskAssignment.user_id',
    backref='assignee',
    lazy=True
    )


    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'name': self.name,
            'phone': self.phone,
            'department': self.department,
            'role': self.role,
            'status': self.status,
            'dark_mode': self.dark_mode,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

# Company Model for CRM - ENHANCED
class Company(db.Model):
    __tablename__ = 'companies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False)
    industry = db.Column(db.String(100))
    size = db.Column(db.String(50))
    website = db.Column(db.String(200))
    description = db.Column(db.Text)
    tags = db.Column(db.String(500))  # NEW: Tags for companies
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'industry': self.industry,
            'size': self.size,
            'website': self.website,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# CRM Contact Model - ENHANCED
class CRMContact(db.Model):
    __tablename__ = 'crm_contacts'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))
    company = db.Column(db.String(200))
    position = db.Column(db.String(100))
    tags = db.Column(db.String(500))
    status = db.Column(db.String(50), nullable=True)
    source = db.Column(db.String(100), default='manual')
    assigned_to = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    deals = db.relationship('Deal', backref='contact', lazy=True)
    tasks = db.relationship('Task', backref='contact', lazy=True)
    activities = db.relationship('Activity', backref='contact', lazy=True)
    comments = db.relationship('Comment', backref='contact', lazy=True)
    attachments = db.relationship('Attachment', backref='contact', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'company': self.company,
            'position': self.position,
            'tags': self.tags,
            'status': self.status,
            'source': self.source,
            'assigned_to': self.assigned_to,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Deal Model - ENHANCED with tags
class Deal(db.Model):
    __tablename__ = 'deals'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200))
    value = db.Column(db.Float, default=0.0)
    stage = db.Column(db.String(50), default='lead')
    probability = db.Column(db.Integer, default=0)
    contact_id = db.Column(db.Integer, db.ForeignKey('crm_contacts.id'))
    assigned_to = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    expected_close = db.Column(db.Date)
    tags = db.Column(db.String(500))  # ENHANCED: Tags for deals
    description = db.Column(db.Text)
    created_date = db.Column(db.Date, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    activities = db.relationship('Activity', backref='deal', lazy=True)
    comments = db.relationship('Comment', backref='deal', lazy=True)
    attachments = db.relationship('Attachment', backref='deal', lazy=True)
    tasks = db.relationship('Task', backref='deal', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'company': self.company,
            'value': self.value,
            'stage': self.stage,
            'probability': self.probability,
            'contact_id': self.contact_id,
            'assigned_to': self.assigned_to,
            'expected_close': self.expected_close.isoformat() if self.expected_close else None,
            'tags': self.tags,
            'description': self.description,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Task Model - ENHANCED with assignment and mentions
class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    due_date = db.Column(db.Date)
    due_time = db.Column(db.Time)
    priority = db.Column(db.String(20), default='medium')
    assigned_to = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    contact_id = db.Column(db.Integer, db.ForeignKey('crm_contacts.id'))
    deal_id = db.Column(db.Integer, db.ForeignKey('deals.id'))
    completed = db.Column(db.Boolean, default=False)
    completed_date = db.Column(db.DateTime)
    tags = db.Column(db.String(500))  # ENHANCED: Tags for tasks
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Calendar fields
    event_type = db.Column(db.String(20), default='task')
    location = db.Column(db.String(200))
    is_all_day = db.Column(db.Boolean, default=False)
    reminder_minutes = db.Column(db.Integer, default=30)
    # NEW: Mentions for tagging users
    mentions = db.Column(db.String(500))  # Store mentioned user IDs

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'due_time': self.due_time.isoformat() if self.due_time else None,
            'priority': self.priority,
            'assigned_to': self.assigned_to,
            'contact_id': self.contact_id,
            'deal_id': self.deal_id,
            'completed': self.completed,
            'completed_date': self.completed_date.isoformat() if self.completed_date else None,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'event_type': self.event_type,
            'location': self.location,
            'is_all_day': self.is_all_day,
            'reminder_minutes': self.reminder_minutes,
            'mentions': self.mentions
        }

# Task Assignment Model - NEW
class TaskAssignment(db.Model):
    __tablename__ = 'task_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    assigned_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    due_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    status = db.Column(db.String(20), default='assigned')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'assigned_by': self.assigned_by,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'notes': self.notes,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Activity Model (Audit Log) - ENHANCED
class Activity(db.Model):
    __tablename__ = 'activities'
    
    id = db.Column(db.Integer, primary_key=True)
    contact_id = db.Column(db.Integer, db.ForeignKey('crm_contacts.id'))
    deal_id = db.Column(db.Integer, db.ForeignKey('deals.id'))
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    entity_type = db.Column(db.String(50))  # NEW: contact, deal, task, company
    entity_id = db.Column(db.Integer)  # NEW: ID of the entity
    old_values = db.Column(db.Text)  # NEW: JSON string of old values
    new_values = db.Column(db.Text)  # NEW: JSON string of new values
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'contact_id': self.contact_id,
            'deal_id': self.deal_id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'action': self.action,
            'description': self.description,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'old_values': json.loads(self.old_values) if self.old_values else None,
            'new_values': json.loads(self.new_values) if self.new_values else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Attachment Model - ENHANCED
class Attachment(db.Model):
    __tablename__ = 'attachments'
    
    id = db.Column(db.Integer, primary_key=True)
    contact_id = db.Column(db.Integer, db.ForeignKey('crm_contacts.id'))
    deal_id = db.Column(db.Integer, db.ForeignKey('deals.id'))
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'))
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(100))
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'contact_id': self.contact_id,
            'deal_id': self.deal_id,
            'task_id': self.task_id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'uploaded_by': self.uploaded_by,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'description': self.description
        }

# Comment Model - ENHANCED with mentions
class Comment(db.Model):
    __tablename__ = 'comments'
    
    id = db.Column(db.Integer, primary_key=True)
    contact_id = db.Column(db.Integer, db.ForeignKey('crm_contacts.id'))
    deal_id = db.Column(db.Integer, db.ForeignKey('deals.id'))
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    mentions = db.Column(db.String(500))  # NEW: Store mentioned user IDs
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'contact_id': self.contact_id,
            'deal_id': self.deal_id,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'text': self.text,
            'mentions': self.mentions,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Notification Model - ENHANCED
class Notification(db.Model):
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), default='info')
    read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    link = db.Column(db.String(500))
    category = db.Column(db.String(50), default='system')
    is_urgent = db.Column(db.Boolean, default=False)
    # NEW: For task assignments and mentions
    related_entity_type = db.Column(db.String(50))  # task, deal, contact
    related_entity_id = db.Column(db.Integer)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'type': self.type,
            'read': self.read,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'link': self.link,
            'category': self.category,
            'is_urgent': self.is_urgent,
            'related_entity_type': self.related_entity_type,
            'related_entity_id': self.related_entity_id
        }

# Reminder Model - NEW
class Reminder(db.Model):
    __tablename__ = 'reminders'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    remind_at = db.Column(db.DateTime, nullable=False)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'remind_at': self.remind_at.isoformat() if self.remind_at else None,
            'is_completed': self.is_completed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# System Settings Model
class SystemSetting(db.Model):
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    description = db.Column(db.String(500))
    category = db.Column(db.String(50), default='general')
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'category': self.category,
            'updated_by': self.updated_by,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# AI Chat History Model
class AIChatHistory(db.Model):
    __tablename__ = 'ai_chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default='general')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'question': self.question,
            'answer': self.answer,
            'category': self.category,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# AI Insights Model
class AIInsight(db.Model):
    __tablename__ = 'ai_insights'
    
    id = db.Column(db.Integer, primary_key=True)
    insight_type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    data = db.Column(db.Text)
    confidence = db.Column(db.Float, default=0.0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'insight_type': self.insight_type,
            'title': self.title,
            'description': self.description,
            'data': json.loads(self.data) if self.data else {},
            'confidence': self.confidence,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

# Bulk Message Campaign Model - NEW
class BulkMessageCampaign(db.Model):
    __tablename__ = 'bulk_message_campaigns'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    message_type = db.Column(db.String(20), nullable=False)  # email, whatsapp
    subject = db.Column(db.String(300))
    message = db.Column(db.Text, nullable=False)
    contact_ids = db.Column(db.Text)  # JSON array of contact IDs
    filters = db.Column(db.Text)  # JSON filter criteria
    sent_count = db.Column(db.Integer, default=0)
    total_contacts = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='draft')  # draft, sending, completed, failed
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sent_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'message_type': self.message_type,
            'subject': self.subject,
            'message': self.message,
            'contact_ids': json.loads(self.contact_ids) if self.contact_ids else [],
            'filters': json.loads(self.filters) if self.filters else {},
            'sent_count': self.sent_count,
            'total_contacts': self.total_contacts,
            'status': self.status,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None
        }

# Export History Model - NEW
class ExportHistory(db.Model):
    __tablename__ = 'export_history'
    
    id = db.Column(db.Integer, primary_key=True)
    export_type = db.Column(db.String(50), nullable=False)  # contacts, companies, deals, tasks, master
    file_format = db.Column(db.String(20), nullable=False)  # excel, pdf
    file_path = db.Column(db.String(500))
    filters = db.Column(db.Text)  # JSON filter criteria
    record_count = db.Column(db.Integer, default=0)
    exported_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exported_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'export_type': self.export_type,
            'file_format': self.file_format,
            'file_path': self.file_path,
            'filters': json.loads(self.filters) if self.filters else {},
            'record_count': self.record_count,
            'exported_by': self.exported_by,
            'exported_at': self.exported_at.isoformat() if self.exported_at else None
        }