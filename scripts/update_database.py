#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update database script for QuangTPS.

This script updates the database schema to handle changes in the application.
It fixes issues like missing columns or table structure changes.
"""

import os
import sys
import logging
import sqlite3
import json
from pathlib import Path

# Add parent directory to path to import quangtps modules
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from quangtps.core.logging import setup_logger
from quangtps.database.db_connector import DBConnector

# Import setup function
sys.path.append(script_dir)
from setup_environment import create_directories

# Set up logger
setup_logger()
logger = logging.getLogger(__name__)

def get_data_dir():
    """Get the data directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    return os.path.join(root_dir, "data")

def update_database_schema():
    """Update the database schema to match the current application version."""
    # Make sure environment is set up
    print("Setting up environment...")
    # Tạo thư mục
    create_directories()
    
    # Get data directory where database is stored
    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, 'database', 'quangtps.db')
    print(f"Checking database at: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        print(f"Database file not found at {db_path}")
        return False
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if patients table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if not cursor.fetchone():
            logger.error("Patients table does not exist")
            print("Patients table does not exist")
            return False
        
        # Get current columns in patients table
        cursor.execute(f"PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Current table columns: {', '.join(columns)}")
        
        # Add missing columns with appropriate defaults
        updates_performed = False
        
        # Check for metadata column
        if 'metadata' not in columns:
            logger.info("Adding metadata column to patients table")
            cursor.execute("ALTER TABLE patients ADD COLUMN metadata TEXT DEFAULT '{}'")
            updates_performed = True
        
        # Fix date of birth column
        if 'birth_date' in columns and 'dob' not in columns:
            logger.info("Renaming birth_date column to dob")
            # SQLite doesn't directly support column rename, so we need to use a workaround
            # Create temporary table
            cursor.execute("CREATE TABLE patients_temp AS SELECT * FROM patients")
            
            # Drop old table
            cursor.execute("DROP TABLE patients")
            
            # Create new table with correct schema
            cursor.execute("""
                CREATE TABLE patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dob TEXT,
                    gender TEXT,
                    address TEXT,
                    phone TEXT,
                    email TEXT,
                    diagnosis TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Get columns from temp table
            cursor.execute("PRAGMA table_info(patients_temp)")
            temp_columns = [col[1] for col in cursor.fetchall()]
            
            # Map birth_date to dob if it exists
            column_mapping = {col: col for col in temp_columns}
            if 'birth_date' in temp_columns:
                column_mapping['birth_date'] = 'dob'
            
            # Build column list for INSERT statement
            source_cols = ', '.join(temp_columns)
            target_cols = ', '.join([column_mapping[col] for col in temp_columns])
            
            # Copy data from temp table to new table
            cursor.execute(f"INSERT INTO patients ({target_cols}) SELECT {source_cols} FROM patients_temp")
            
            # Drop temp table
            cursor.execute("DROP TABLE patients_temp")
            updates_performed = True
        
        # Create dob column if neither birth_date nor dob exists
        if 'dob' not in columns and 'birth_date' not in columns:
            logger.info("Adding dob column to patients table")
            cursor.execute("ALTER TABLE patients ADD COLUMN dob TEXT")
            updates_performed = True
        
        # Commit changes
        conn.commit()
        
        if updates_performed:
            logger.info("Database schema updated successfully")
            print("Database schema updated successfully")
        else:
            logger.info("No database schema updates needed")
            print("No database schema updates needed")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating database schema: {str(e)}", exc_info=True)
        print(f"Error updating database schema: {str(e)}")
        if conn:
            conn.rollback()
        return False
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("QuangTPS Database Update Tool")
    print("----------------------------")
    update_database_schema()
    print("Database update completed.") 