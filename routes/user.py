from flask import Blueprint, request, jsonify
from utils.password_utils import verify_password, hash_password
from db import get_db_connection   # This should return your MongoDB Database object
from typing import Any, Dict, Optional

user_bp = Blueprint("user", __name__)

@user_bp.route("/change-password", methods=["POST"])
def change_password():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_id = data.get("user_id")
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    # In your project, 'db' is likely a MongoDB Database object
    db: Any = get_db_connection()
    
    # Access the 'users' collection using square brackets (Safe & Pylance-friendly)
    # We find the user by their ID (or email/username depending on your schema)
    user: Optional[Dict[str, Any]] = db["users"].find_one({"user_id": user_id})

    if not user:
        return jsonify({"error": "User not found"}), 404

    # MongoDB returns a dictionary, so we access the hashed password directly
    hashed_password_in_db = str(user.get("password") or "")

    if not verify_password(hashed_password_in_db, str(old_password or "")):
        return jsonify({"error": "Old password incorrect"}), 400

    # Hash the new password and update the document
    new_hashed = hash_password(str(new_password or ""))
    
    db["users"].update_one(
        {"user_id": user_id},
        {"$set": {"password": new_hashed}}
    )

    return jsonify({"message": "Password updated successfully"})