# models.py - Database Models
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timedelta
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with scans
    scans = db.relationship('NutritionScan', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class NutritionScan(db.Model):
    __tablename__ = 'nutrition_scans'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_name = db.Column(db.String(200), default='Unknown Product')
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    
    # Store nutrients as JSON
    nutrients_data = db.Column(db.Text, nullable=False)  # JSON string
    
    def __repr__(self):
        return f'<NutritionScan {self.id} - {self.product_name}>'
    
    @property
    def nutrients(self):
        """Get nutrients as Python object"""
        try:
            return json.loads(self.nutrients_data)
        except (json.JSONDecodeError, TypeError):
            return []
    
    @nutrients.setter
    def nutrients(self, value):
        """Set nutrients from Python object"""
        self.nutrients_data = json.dumps(value)
    
    @property
    def is_expired(self):
        """Check if scan has expired (older than 1 month)"""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self):
        """Convert scan to dictionary for API responses"""
        return {
            'id': self.id,
            'product_name': self.product_name,
            'scan_date': self.scan_date.isoformat(),
            'nutrients': self.nutrients,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }