FLAG 2027 - encoder fine-tuning on your own PC (RTX 4070)
=========================================================

What this does: the same cross-modal fine-tuning that is running on Kaggle (VoxCeleb2 only, Bengali speakers
removed, no FLAG training data; two variants sg50 / sg90), but on this PC's GPU with no 12-hour limit.

1. Unzip this folder somewhere with at least 25 GB free, e.g. C:\flagft
2. Get a Kaggle API token for the shahiraalima account:
   kaggle.com (logged in as shahiraalima) -> Settings -> API -> Generate New Token -> copy it.
   The scripts ask for it; it is only kept for that PowerShell window, never saved to disk.
3. Open PowerShell in the folder and run:
      powershell -ExecutionPolicy Bypass -File .\setup.ps1
   (installs Python 3.11 if needed, PyTorch with CUDA, downloads ~10 GB of data; ends with "SETUP COMPLETE")
4. Then:
      powershell -ExecutionPolicy Bypass -File .\run.ps1
   Keep the window open and the PC awake. After a few minutes it prints lines like
      [1 encoders unfrozen] step 100/4000 0.45 s/step
   Send Claude one of those lines and you will get a time estimate.
   If it says "CUDA out of memory":  powershell -ExecutionPolicy Bypass -File .\run.ps1 -P 48
5. When it says FINISHED:
      powershell -ExecutionPolicy Bypass -File .\upload.ps1
   (uploads the results as a PRIVATE Kaggle dataset; tell Claude when it is done)

Privacy: the FLAG data must stay private. When we are done with this PC, delete the "input" folder
(and "venv"), and revoke the Kaggle token on kaggle.com.
