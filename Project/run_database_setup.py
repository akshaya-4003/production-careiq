"""Entry point 2: create the database and seed one account per role.

    python3 run_database_setup.py

Safe to run more than once: existing tables are left alone and an account that
already exists is not touched.
"""

import sys

from config import config
from persistence.database import create_all_tables, create_database_directory, engine, \
    open_database_session
from persistence.repository import count_users, find_user, insert_user
from security.passwords import PasswordVault


def main():
    folder = create_database_directory()
    tables = create_all_tables(engine)
    print("Database directory ready: " + folder)
    print("Database file           : " + config.APPLICATION_DATABASE_FILE)
    print("Tables available        : " + str(len(tables)))
    for table in tables:
        print("    - " + table)

    vault = PasswordVault()
    session = open_database_session()
    try:
        seeded = []
        for account in config.DEMO_USERS:
            if not find_user(session, account["user_name"]):
                insert_user(session, {
                    "user_name": account["user_name"],
                    "display_name": account["display_name"],
                    "password_hash": vault.hash_password(account["password"]),
                    "role": account["role"],
                })
            seeded.append(account["user_name"])
        print("Seeded demo users       : " + ", ".join(seeded))
        print("Sign in with any of these:")
        for account in config.DEMO_USERS:
            print("    {0:<8} / {1:<14} role: {2}".format(
                account["user_name"], account["password"], account["role"]))
        print("Users in the database   : " + str(count_users(session)))
    finally:
        session.close()
    print("Done. Next: python3 run_data_preparation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
