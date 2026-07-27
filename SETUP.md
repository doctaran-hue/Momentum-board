# Setup by pasting — no uploads

Three files. Create each one inside GitHub's web editor. Typing a path with
slashes in the filename box creates the folders for you.

> Why this way: GitHub's drag-and-drop uploader silently skips folders that
> start with a dot, so `.github/workflows/` never arrives and nothing ever runs
> on a schedule. Typing the path in the **Create new file** box has no such
> problem.

## 0. Make the repo

github.com → **New repository** → name it `momentum-board` → **Public**
(Actions and Pages are free on public repos) → tick **Add a README** →
Create repository.

## 1. scan.py

**Add file → Create new file**. Filename: `scan.py`
Paste the contents of `scan.py`. **Commit changes.**

## 2. docs/index.html

**Add file → Create new file**. Filename: type exactly

    docs/index.html

GitHub turns that into a folder as you type the slash. Paste the contents of
`index.html`. **Commit changes.**

## 3. The scheduler

Easiest route is the Actions tab, which puts you straight into the right folder:

**Actions** → *set up a workflow yourself* → replace everything in the editor
with the contents of `daily.yml` → rename the file to `daily.yml` →
**Commit changes.**

Or: **Add file → Create new file**, filename `.github/workflows/daily.yml`.

## 4. Two settings

- **Settings → Pages** → Source: *Deploy from a branch*, Branch `main`,
  Folder `/docs` → Save
- **Settings → Actions → General** → Workflow permissions →
  **Read and write permissions** → Save

Without the second one the job runs but cannot save its results.

## 5. First run

**Actions → daily scan → Run workflow.** Takes 5–10 minutes.

Your board: `https://<username>.github.io/momentum-board/`

Until the first run finishes the page says "No scan data yet" — that is the
expected state, not an error. After that it updates itself at 18:10 IST every
weekday and you never touch it again.

## Editing later

`scan.py` has a knobs block near the top — RS threshold, turnover floor, gate
thresholds. Edit it directly on GitHub (open the file, pencil icon, commit) and
the next run picks it up.
