import sys
from pathlib import Path

# Ensure backend package is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import create_app  # noqa: E402
from app.db import db  # noqa: E402
from app.models import UserBankAccount, UserBankDocument  # noqa: E402,F401


def main():
    app = create_app()
    with app.app_context():
        engine = db.engine
        for table in (UserBankAccount.__table__, UserBankDocument.__table__):
            if not engine.dialect.has_table(engine.connect(), table.name):
                table.create(bind=engine)
                print(f"✅ Created table: {table.name}")
            else:
                print(f"ℹ️ Table already exists: {table.name}")


if __name__ == "__main__":
    main()

