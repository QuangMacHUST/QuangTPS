#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to update the database schema for QuangTPS.
This script ensures that the database has the necessary structure for
the current version of the application.
"""

import os
import sys
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from quangtps.database.db_connector import DBConnector
from quangtps.core.config import Config
from quangtps.core.logging import setup_logger, get_logger

def setup_logging():
    """Set up logging configuration."""
    setup_logger(level='INFO')
    return get_logger(__name__)

def update_database_structure(logger):
    """Update database schema to latest version."""
    logger.info("Updating database structure...")
    db = DBConnector.get_instance()
    conn = db.connection()
    
    try:
        # Check if birth_date column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(patients)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add birth_date column if it doesn't exist
        if 'birth_date' not in columns:
            logger.info("Adding 'birth_date' column to patients table...")
            cursor.execute("ALTER TABLE patients ADD COLUMN birth_date TEXT")
            
            # Update existing records to copy data from dob to birth_date
            if 'dob' in columns:
                logger.info("Copying data from 'dob' to 'birth_date'...")
                cursor.execute("UPDATE patients SET birth_date = dob WHERE dob IS NOT NULL")
        
        # Add dob column if it doesn't exist
        if 'dob' not in columns:
            logger.info("Adding 'dob' column to patients table...")
            cursor.execute("ALTER TABLE patients ADD COLUMN dob TEXT")
            
            # Update existing records to copy data from birth_date to dob
            if 'birth_date' in columns:
                logger.info("Copying data from 'birth_date' to 'dob'...")
                cursor.execute("UPDATE patients SET dob = birth_date WHERE birth_date IS NOT NULL")
        
        # Commit changes
        conn.commit()
        logger.info("Database structure successfully updated.")
        
        # Create update entry
        update_note = {
            "version": "1.0.1",
            "timestamp": datetime.now().isoformat(),
            "description": "Updated database to support both 'birth_date' and 'dob' fields for patient dates"
        }
        
        # Check if updates table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='database_updates'")
        if not cursor.fetchone():
            logger.info("Creating database_updates table...")
            cursor.execute("""
                CREATE TABLE database_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    applied BOOLEAN DEFAULT 1
                )
            """)
        
        # Add update record
        cursor.execute(
            "INSERT INTO database_updates (version, timestamp, description) VALUES (?, ?, ?)",
            (update_note["version"], update_note["timestamp"], update_note["description"])
        )
        conn.commit()
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Error updating database: {e}", exc_info=True)
        conn.rollback()
        return False
    
    return True

def verify_database_updates(logger):
    """Verify that database updates were applied correctly."""
    logger.info("Verifying database updates...")
    db = DBConnector.get_instance()
    conn = db.connection()
    
    try:
        cursor = conn.cursor()
        
        # Verify both columns exist
        cursor.execute("PRAGMA table_info(patients)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'birth_date' in columns and 'dob' in columns:
            logger.info("Verification successful: both 'birth_date' and 'dob' columns exist.")
            return True
        else:
            missing = []
            if 'birth_date' not in columns:
                missing.append('birth_date')
            if 'dob' not in columns:
                missing.append('dob')
            logger.error(f"Verification failed: missing columns {', '.join(missing)}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying database: {e}", exc_info=True)
        return False

def backup_database(logger, db_path):
    """Create a backup of the database before making changes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{db_path}.backup_{timestamp}"
    
    try:
        logger.info(f"Creating database backup: {backup_file}")
        # Copy database file
        with open(db_path, 'rb') as src, open(backup_file, 'wb') as dst:
            dst.write(src.read())
        logger.info("Backup created successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}", exc_info=True)
        return False

def main():
    """Main function to run the database update script."""
    parser = argparse.ArgumentParser(description='Update QuangTPS database schema.')
    parser.add_argument('--verify-only', action='store_true', help='Only verify the database schema without making changes')
    parser.add_argument('--skip-backup', action='store_true', help='Skip database backup')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting database update process...")
    
    # Get database path
    config = Config.get_instance()
    db_dir = os.path.join(config.data_dir, 'database')
    db_path = os.path.join(db_dir, 'quangtps.db')
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return 1
    
    # Verify only mode
    if args.verify_only:
        if verify_database_updates(logger):
            logger.info("Database schema verification: PASSED")
            return 0
        else:
            logger.error("Database schema verification: FAILED")
            return 1
    
    # Create backup unless skipped
    if not args.skip_backup:
        if not backup_database(logger, db_path):
            logger.error("Database backup failed, aborting update.")
            return 1
    
    # Update database
    if update_database_structure(logger):
        logger.info("Database update completed successfully.")
        
        # Verify after update
        if verify_database_updates(logger):
            logger.info("Database verification after update: PASSED")
            return 0
        else:
            logger.error("Database verification after update: FAILED")
            return 1
    else:
        logger.error("Database update failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 