from flask import Blueprint, request, jsonify
from utils.password_utils import verify_password, hash_password
from db import get_db_connection   # use your existing DB connection

user_bp = Blueprint("user", __name__)

@user_bp.route("/change-password", methods=["POST"])
def change_password():
    data = request.json
    user_id = data.get("user_id")
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT password FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not verify_password(user["password"], old_password):
        return jsonify({"error": "Old password incorrect"}), 400

    new_hashed = hash_password(new_password)
    cursor.execute(
        "UPDATE users SET password=%s WHERE id=%s",
        (new_hashed, user_id)
    )
    conn.commit()

    return jsonify({"message": "Password updated successfully"})
