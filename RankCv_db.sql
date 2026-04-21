-- Create Database
CREATE DATABASE IF NOT EXISTS rankcv_db;
USE rankcv_db;

-- Users Table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    profile_pic VARCHAR(255) NULL,
    subscription_plan VARCHAR(20) DEFAULT 'free',
    subscription_expires DATETIME NULL,
    stripe_customer_id VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Usage Logs Table
CREATE TABLE usage_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_date DATE NOT NULL,
    sessions_used INT DEFAULT 0,
    cvs_processed INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_date (user_id, session_date)
);

-- Settings Table (UPDATED with unit columns)
CREATE TABLE user_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNIQUE NOT NULL,
    experience_enabled BOOLEAN DEFAULT TRUE,
    min_experience INT DEFAULT 0,
    min_experience_unit VARCHAR(10) DEFAULT 'years',
    max_experience INT DEFAULT 10,
    max_experience_unit VARCHAR(10) DEFAULT 'years',
    show_score_bar BOOLEAN DEFAULT TRUE,
    save_cvs BOOLEAN DEFAULT FALSE,
    cv_limit INT DEFAULT 20,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Ranking History Table
CREATE TABLE ranking_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cv_count INT DEFAULT 0,
    top_score DECIMAL(5,2) DEFAULT 0,
    top_candidate VARCHAR(255),
    results_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Insert default admin/test user (optional)
INSERT INTO users (email, subscription_plan) 
VALUES ('test@example.com', 'free');