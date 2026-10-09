# Marketing Decision Lab update

Replace app.py in your existing GitHub repository and commit. Other source files and datasets remain compatible.

Overview now provides six business questions, findings, next actions and evidence limits. Headlines use currently loaded data; customer segmentation uses a three-cluster default scenario. CLV overview uses explicitly stated default assumptions.

## Optional AI report

The automatic decision summary works without a key. To enable genuine AI narrative generation, add the following through Streamlit app settings → Secrets, never in GitHub:

```toml
OPENAI_API_KEY = "your-private-api-key"
OPENAI_MODEL = "gpt-4.1"
```

Use a model available to your account. AI calls occur only on pressing Generate AI decision report. They send the six aggregate findings, dataset source categories and campaign assumptions, not row-level data, customer IDs, filenames or CSV files. Responses use store=false. Changes to the input findings or report language hide stale reports. Provider calls require a configured account and can incur charges. No live provider call was tested without credentials; the offline report and page navigation were tested locally.
