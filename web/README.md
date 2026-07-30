# Deployed browser application

This directory contains the reader-facing source used by the deployed
application. The model payload is generated rather than committed:

```bash
python scripts/export_browser_model.py \
  --output web/public/model.json.gz
```

The generated file contains the fitted preprocessor state and the 100
depth-capped decision trees. It is excluded from source control because it is a
derived binary artifact. The application decompresses it, reproduces the
scikit-learn transformations, traverses every tree, and averages the tree
predictions in the browser.

The production application is available at
[crop-yield-intelligence.alutiba.chatgpt.site](https://crop-yield-intelligence.alutiba.chatgpt.site).
