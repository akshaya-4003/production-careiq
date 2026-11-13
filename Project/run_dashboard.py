"""Entry point 7: start the dashboard on port 8501.

    python3 -m streamlit run run_dashboard.py

The dashboard is a client of the REST service; start python3 run_api_service.py
first. With the service down it says so and names the command that starts it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.pages import main  # noqa: E402

main()
