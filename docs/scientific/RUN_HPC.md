# 🧬 Running Scientific Multimodal RAG on IITJ HPC Cluster

This guide explains how to deploy, queue, monitor, and tunnel the RAG application on the IIT Jodhpur HPC cluster using the SLURM scheduler.

---

## 🚫 Critical HPC Cluster Restrictions

Unlike a local computer, you cannot run model-heavy applications directly on the login node (`172.25.0.81`). 
* **No `srun`**: Interactive allocation commands (e.g., `srun ... --pty bash`) are disabled on the login node to protect resource limits.
* **Must use Batch Jobs (`sbatch`)**: To run the application, you must submit it as a background job to the cluster.
* **Memory Limits**: The cluster rejects jobs that do not request a minimum of 1 GB RAM per CPU core (must use `--mem-per-cpu`).

---

## 🏃 Step-by-Step Execution Guide

### 1. Log in and navigate to the project directory
Connect to the login node:
```bash
ssh divyasaxena_rs@172.25.0.81
```
Navigate to your workspace folder:
```bash
cd /scratch/data/divyasaxena_rs/Vineet_internship
```
* **Why?** Files and execution scripts must reside under the shared `/scratch/` mount so compute nodes can read and write to them.

---

### 2. Submit the Job to SLURM
Submit your script to the batch queue scheduler:
```bash
sbatch scripts/slurm_app.sh
```
* **Why use this command?** This requests a GPU node from the cluster. It will output: `Submitted batch job <JOB_ID>` (e.g., `346287`).
* **Why did we change the partition to `fat`?** The `dgx` GPU partition is often congested, meaning jobs can queue for 2+ days. The `fat` partition contains nodes (like `cn22` and `cn25`) with multiple free GPUs, starting your job instantly.

---

### 3. Check the Job Status
Check the status of your queued jobs:
```bash
squeue -u divyasaxena_rs
```
* **Why use this command?** This queries the SLURM scheduler to check the state of your job under your user account:
  * **`PD` (Pending)**: The job is waiting for a node allocation.
  * **`R` (Running)**: The job is currently active. Note the **Node Name** under the `NODELIST` column (e.g., `cn27`). You will need this node name for port forwarding!

---

### 4. View Logs in Real-Time (Tailing)
Once the job is in the **`R` (Running)** state, you can monitor the terminal output:
```bash
tail -f /scratch/data/divyasaxena_rs/sci_rag_app_<JOB_ID>.out
```
* **Why use this command?** Because the job runs in the background, stdout/stderr is written to a file. The `tail -f` (follow) command reads and prints new lines instantly as they are logged.
* **Note**: If the job is still pending (`PD`), running this command will return `No such file or directory` because SLURM only creates the file once execution starts.

---

### 5. Establish SSH Port Forwarding Tunnel
To open the Streamlit web interface on your laptop, you must tunnel the network port.

Open a **new, local terminal window on your laptop** (do not run this inside your active SSH session) and run:
```bash
ssh -L 8501:<NODE_NAME>:8501 divyasaxena_rs@172.25.0.81
```
* *Example (if running on `cn27`):* `ssh -L 8501:cn27:8501 divyasaxena_rs@172.25.0.81`
* *Troubleshooting:* If local port `8501` is already in use, map to `9000`: `ssh -L 9000:cn27:8501 divyasaxena_rs@172.25.0.81`

#### ❓ Why does this command work?
1. **`-L 8501:<NODE_NAME>:8501`**: Maps port `8501` of your local laptop, through the gateway login node (`172.25.0.81`), directly to port `8501` of the compute node (e.g. `cn27`) where your app is running.
2. **Why not use `localhost`?** If you forward to `localhost`, you are forwarding to the login node itself, where the Streamlit app is not running. You must specify the specific hostname of the compute node (like `cn27`) where your job was allocated.

---

### 6. Open the Application
Once the tunnel is active, open your laptop browser and go to:
* **`http://localhost:8501`** (or `http://localhost:9000` if you changed the local port).
