from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash

change_password_bp = Blueprint("change_password", __name__)

@change_password_bp.route("/change_password", methods=["POST"])
@login_required
def change_password():
    data = request.get_json()

    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not old_password or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    user = current_user

    # verify old password
    from werkzeug.security import check_password_hash
    if not check_password_hash(user.password, old_password):
        return jsonify({"error": "Old password is incorrect"}), 401

    # update password
    user.password = generate_password_hash(new_password)

    return jsonify({"message": "Password changed successfully"}), 200
