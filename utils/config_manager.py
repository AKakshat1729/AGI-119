import os

def update_env_variable(key: str, value: str):
    """
    Updates or adds a key-value pair in the .env file.
    """
    env_path = os.path.join(os.getcwd(), '.env')
    lines = []
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            
    found = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
            
    if not found:
        # Add newline if last line doesn't have one
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        new_lines.append(f"{key}={value}\n")
        
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    # Also update current environment
    os.environ[key] = value
