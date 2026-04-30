# 🚀 ORIX — KAUST Cloud GPU Cluster

*Named after the Arabian oryx—built for endurance, speed, and thriving under pressure.*

Welcome to ORIX—your (temporary) superpower.

> **NOTE:** This website is still under construction. Expect changes in the upcoming days.

Through the VPR Office, KAUST is making available a cloud-hosted GPU cluster for a limited period of 6 months, featuring:

- **128× GPUs total:** 64× NVIDIA H100 plus 64× NVIDIA H200 GPUs (in these pages we will use the shorthand term HGPU to denote a GPU based on the NVIDIA Hopper architecture)

This is a high-end, high-demand, and very real shared system. Treat it well—and it will treat your experiments even better.

## 🎯 What This Is (and Isn't)

This cluster is:

- ✅ Exclusively for KAUST internal research projects
- ✅ Available to KAUST faculty, students, staff and postdocs
- ✅ Designed for high utilization and real workloads

This cluster is not:

- ❌ A personal playground
- ❌ A backup for poorly scheduled experiments
- ❌ A resource for external collaborations (formal or informal)

Access is granted by request and explicit approval.

## ⚠️ Required Reading (Yes, Really)

Before you run anything:

- 👉 Read the cluster documentation
- 👉 Pay special attention to preemption
- 👉 ORIX is preemption-first by design

**Why?**

- You will run jobs on shared resources
- You will encounter preemption
- You really don't want your 3-day job to disappear without checkpoints

If you skip this step, the cluster will eventually teach you anyway—but less gently.

## 🧠 How the Cluster Thinks

This is a SLURM-based system with:

- Shared unreserved resources
- PI-based reserved allocations
- Opportunistic access via preemptible jobs

**Key idea:** ORIX prioritizes utilization over guarantees. If you're using idle capacity, your job must be preemptible—and therefore disposable.

Design your jobs accordingly:

- Checkpoint early
- Checkpoint often
- Assume nothing is forever

## 🔐 Access & Authentication

- Access is restricted to whitelisted KAUST users only
- Authentication is SSH key–based only (no passwords)
- All activity is logged and auditable

You are responsible for:

- Your SSH keys
- Your jobs
- Your impact on others

To request access, please submit the application form. Each application is reviewed individually.

## 📜 Usage Principles

This cluster exists to maximize throughput, not comfort.

You agree to:

- Use resources responsibly and efficiently
- Respect reservations and quotas
- Use preemptible jobs for opportunistic usage
- Avoid idle reservations (they will be reclaimed)
- Follow all documented policies and best practices

## 🤝 A Shared System

This is a community resource.

**Good citizens:**

- Release resources when done
- Don't hoard GPUs "just in case"
- Write resilient jobs
- Help others avoid mistakes you already made

**Bad citizens:**

- Learn these lessons the hard way

## 🧾 Disclaimer and Acknowledgment

By using this cluster, you understand and agree that:

- Access is granted exclusively for internal KAUST projects
- Resources may not be used for any external collaborations (formal or informal)
- Access is limited to full-time KAUST staff, students, and postdoctoral researchers
- All users must be explicitly whitelisted prior to access
- All access and usage is logged, monitored, and auditable

### Authentication

- Access is provided exclusively via SSH public key authentication
- Password-based authentication is not permitted
- You are responsible for safeguarding your private key
- Any activity performed using your SSH key is attributed to you

### 📊 Usage Policy Acknowledgment

You acknowledge that:

- This is a shared, high-value resource designed to maximize utilization
- The system uses SLURM, requiring familiarity with job submission and management
- Access is governed by reservations and/or quotas assigned to PI groups
- Idle resources may only be accessed via preemptible jobs

You further agree that:

- Preemptible jobs may be interrupted at any time
- You are responsible for handling preemption correctly (including cleanup and checkpointing)
- Reservations must be requested and approved in advance
- Misuse or underutilization of reserved resources may lead to revocation

## 🚦 Before You Start

- ✔ Get access approved
- ✔ Submit your SSH key
- ✔ Read the documentation (Cluster info, GPU cluster policy, Preemptible jobs, Software policy)
- ✔ Check out the FAQ
- ✔ Understand the environment and take the quiz
- ✔ Test small before scaling

## 🧪 Final Advice

Start simple. Scale responsibly. Expect interruption.

ORIX rewards those who read the docs—and humbles those who don't.

Make sure to complete the online quiz to unlock access to ORIX.

---

# Cluster Info

## Cluster Composition

| Node | CPUs | RAM | GPU |
|------|------|-----|-----|
| login-[0-1] | AMD EPYC-Genoa Processor (32 vCPUs) | 126 GiB | — |
| orix-worker-h100-[0-7] | 2× Intel Xeon Platinum 8468 (128 vCPUs) | 1.5 TiB | 8× H100 80GB |
| orix-worker-h200-[0-7] | 2× Intel Xeon Platinum 8468 (128 vCPUs) | 1.5 TiB | 8× H200 141GB |

> **Warning:** The login nodes and GPU worker nodes have different CPUs. While both are x86 architectures, they do not support the same instruction set extensions. For example, the CPU on the GPU nodes supports `amx_bf16`, which is not available on the CPU of the login nodes. Conversely, the login nodes also have instructions that are not available on the workers. Because of these differences, avoid compiling on the login nodes with flags like `-O3 -march=native -mtune=native`. Binaries built on the login node and executed on the GPU nodes may crash with an "illegal instruction" error, or less noticeably, may fail to take advantage of instructions available on the workers, leading to reduced performance.

## Storage

Users have access to the following filesystems:

- `/home/<uname>`: Use it for your development environment and lightweight files.
- `/mnt/data/u/<uname>`: Use this for all data-intensive artifacts such as datasets, checkpoints, and outputs.

As a general rule, keep `/home` lean and move heavy project data to `/mnt/data/u/<uname>`.

Storage capacity is currently 200 TB, with plans to increase it by 200 TB each month until 1 PB (after 5 months).

Per-user storage limits are not currently enforced, but usage is monitored and accounts may be disabled for intentional misuse.

## Connecting to ORIX

Once your access request has been approved, you can connect to ORIX using the command below. Be sure to use your KAUST username along with the SSH key pair shared through the application form. The private key corresponding to the submitted public key is required for authentication.

```bash
ssh <kaust_username>@orix-login.kaust.edu.sa
```

This will connect you to one of the login nodes, where you can configure your environment and submit jobs.

To simplify the access command, you can add an entry to your SSH config file (`~/.ssh/config`):

```
Host orix
  User <kaust_username>
  Hostname orix-login.kaust.edu.sa
  IdentityFile <path_to_ssh_private_key>
```

After that, you can connect simply with:

```bash
ssh orix
```

---

# Cluster Policy

## Overview

Cluster resources are organized into two tiers:

- Shared unreserved resources
- PI-based reserved resources

Most users will interact with the GPU cluster using the unreserved resources tier. All users have equal priority within this tier, subject to usage limits (e.g. maximum number of GPUs used at once).

In addition to the unreserved resources, certain PI groups receive prioritized access to a portion of the cluster through reserved resources. These reservations guarantee the group a baseline amount of resources that is independent of the current utilization of the unreserved shared pool, ensuring the group can always access a baseline amount of compute resources when needed.

Reserved resources guarantee resource availability, but they don't enforce exclusive access. When reserved or unreserved resources are unused, this idle capacity can be accessed using preemptible jobs. These jobs can run on any idle resource, with the understanding that they might be terminated if a higher-priority job needs them, for example a job from a PI group trying to use their reserved resources. Another use for preemptible jobs is to access a larger share of the cluster resources when they are idle, as these jobs are associated with more relaxed usage limits (e.g. allow access to more GPUs).

In summary:

- Any user can access unreserved resources
- Target PI groups can expect guaranteed resources through reservations
- All idle resources can be used through preemptible jobs
- Preemptible jobs have more relaxed usage limits than jobs on unreserved resources

## Job Scheduling

- Job scheduling is based on the job's priority. Higher priority jobs are scheduled first.
- Users who have not been running jobs recently receive a small priority boost.
- The Fairshare window is two weeks.
- Jobs configured as preemptible can utilize any idle resource.
- Preemptible jobs have the lowest priority.
- Lower-priority jobs can run before a higher-priority job if they don't delay its starting time.
- Non-preemptible jobs can evict jobs configured as preemptible.
- Nodes are shared. Multiple jobs can run on the same node at the same time.

## Available Partitions

> Final values are still being decided.

| Partition | Resources | Description |
|-----------|-----------|-------------|
| batch-h100 | 24× H100 | Unreserved pool of nodes |
| batch-h200 | 24× H200 | Unreserved pool of nodes |
| pi-\<uname\> | 4,8× HGPU | Reserved resources for a PI group |
| freecycle | 128× HGPU | Only usable with preemptible jobs |

*pi-\<uname\> = PI - username*

A small interactive reservation for debugging is being considered.

## Job Priorities and QoS

A job's queueing priority is mostly determined by its quality of service (QoS). The QoS also enforces resource limits per PI group as well as maximum job run times. Jobs submitted to a partition default to using the QoS of the same name.

> Final values are still being decided.

| QoS | PI Group Limits | User Limits | Max Time |
|-----|-----------------|-------------|----------|
| batch | 8 HGPU, 128 CPU, 1500 GB | — | 3 days |
| pi-\<uname\> (half-node) | 4 HGPU, 64 CPU, 750 GB | — (PI-conf) | 3 days (PI-conf) |
| pi-\<uname\> (full-node) | 8 HGPU, 128 CPU, 1500 GB | — (PI-conf) | 3 days (PI-conf) |
| freecycle | 32 HGPU, 1024 CPU, 750 GB | — | 3 days |

*pi-\<uname\> = PI - username*

By default any user from a PI group can use all the group's resources.

The cluster keeps time limits relatively short in order to encourage users to implement checkpointing mechanisms, making their jobs suitable for execution as preemptible jobs in the freecycle partition. Using it allows users to take advantage of large amounts of idle capacity when they become temporarily available.

PIs can customize the QoS for their reservation with reduced per-user limits or different time limits.

The freecycle QoS can be preempted by every other QoS and has 10× less priority in the queue.

## Does My PI Have Reserved Resources?

The easiest way to find out is to attempt to use reserved resources. See the example cluster usage below.

## Example Cluster Usage

The above partitions and QoSes can be directly passed to the `srun` command and `sbatch` scripts. For convenience, short aliases are also available to avoid the need to specify the common partition/QoS combinations.

```bash
# Unreserved resources
srun | sbatch
## (defaults to --partition=batch-h100 --qos=batch)
## To target the h200 partition:
srun | sbatch --partition=batch-h200

# Reserved resources (Replace `uname` with your PI's KAUST username)
srun | sbatch --partition=pi-<uname> --qos=pi-<uname>
# or
srun-pi | sbatch-pi

# Preemptible job
srun | sbatch --partition=freecycle --qos=freecycle
# or
srun-free | sbatch-free
```

To test the short aliases, try these commands:

```bash
$ srun      --gpus=1 --time=1:00 -- nvidia-smi
$ srun-pi   --gpus=1 --time=1:00 -- nvidia-smi
$ srun-free --gpus=1 --time=1:00 -- nvidia-smi
```

---

# Job Preemption

Job preemption is the act of canceling "lower priority" jobs to let a "higher priority" job run. When a job that can preempt others is allocated resources that are already allocated to one or more running jobs that can be preempted, the lower priority job(s) are stopped, in other words, preempted. Not all jobs are at risk of preemption. Only jobs that request using specific partitions can be preempted.

## Suitable Workloads for Preemptible Jobs

Preemptible jobs have larger resource limits than regular jobs, allowing users to access additional resources temporarily.

The most suitable workloads for preemptible jobs are those that can checkpoint and resume their state. These jobs will be able to take advantage of the temporary idle resources until getting interrupted without losing progress.

But even workloads without checkpointing can benefit from preemptible jobs as many jobs will be able to run to completion without being interrupted. The longer a job runs for or the more resources it requests, the higher the chance it can get interrupted; so without checkpointing, small jobs using small resources are ideal.

## How to Run a Preemptible Job

Jobs are not preemptible by default. Only jobs using the freecycle partition can be at risk for preemption. To use this partition, the job must also use the freecycle QoS:

```bash
srun | sbatch --partition=freecycle --qos=freecycle
# or using the short form
srun-free | sbatch-free
```

## Requeuing Preempted Jobs

By default, when a job is preempted, it is canceled. If your job can safely be requeued, it can be configured to be automatically requeued by adding the `--requeue` Slurm option in the job script.

On preemption, jobs receive a signal and are granted a **60 second grace period** to gracefully stop running tasks and potentially save or checkpoint the current state so it can be resumed on the next requeue.

Keep in mind that on requeue, the entry script for your job will start from the beginning. If the job wants to resume from a previous checkpoint, it should start by checking if it can continue from a checkpoint. Some machine learning frameworks like PyTorch Lightning already support this.

---

# Software Policy

Unlike traditional HPC systems that use modules to share software, this cluster provides minimal base software. Instead, users are expected to install and manage all their software in their home directory. This decision aims to make users comfortable managing their entire software stack, reduces reliance on centrally managed software, and encourages projects to track all dependencies, improving reproducibility.

There are several tools that facilitate per-user software installation. One of the most widely used is Conda, an open-source package and environment manager. We recommend **Pixi**, a newer tool built on the Conda ecosystem. Pixi uses the same package repositories, but provides a significantly improved user experience.

Regardless of the tool you choose, we recommend it has the following features:

- **Per-project environments** — Different projects have different dependencies so each should have its own environment. Examples:
  - `~/dev/llm-project` (pytorch=2.2.0, transformers=4.49.0, cuda-toolkit=12.8)
  - `~/dev/llm-project-2` (pytorch=2.6.0, transformers=5.1.0)
- **Multi-language support** — Python, C/C++, Rust, Node.js. Python-only tools like uv will not have this flexibility.
- **Large package ecosystem** — Languages & runtimes (python, nodejs), compilers and build tools (g++, clang, cuda-toolkit, cmake), CLI tools (git-lfs, just, ripgrep, jq).

> **Warning:** Please keep in mind that only your development environment should be installed in your home directory. Data-intensive artifacts such as datasets, checkpoints, and outputs must not be stored there. See the storage notes for the recommended locations.

No default CUDA runtime or toolkit are installed.

## Pixi Short Guide

Below is a brief introduction highlighting the ease of use of Pixi. Pixi has many other features and excellent documentation which we recommend exploring in depth. The full documentation including install instructions can be found on Pixi's website.

### Working with Environments

```bash
# Initialize a new project
$ pixi init ml-project
✔ Created /home/user/ml-project/pixi.toml
$ cd ml-project

# Add a specific Python version and compatible pip
$ pixi add python=3.12 pip
✔ Added python=3.12
✔ Added pip >=26.0.1,<27

# Activate the environment in the current dir (ml-project)
# Notice that we don't need to specify the environment name like in conda
$ pixi shell
(ml-project) $

# Install a Python package with pip.
# Python packages can also be installed directly through Pixi.
(ml-project) $ pip install numpy=2.4.3
Successfully installed numpy-2.4.3

# Confirm that installations are under project directory
(ml-project) $ which python
/home/user/ml-project/.pixi/envs/default/bin/python
(ml-project) $ which pip
/home/user/ml-project/.pixi/envs/default/bin/pip
(ml-project) $ pip show numpy
pip 26.0.1 from /home/user/ml-project/.pixi/envs/default/lib/python3.12/site-packages/pip (python 3.12)

# Run a command inside the environment
(ml-project) $ python -c 'print("Hello World!")'
Hello World!

# Exit the environment
(ml-project) $ exit

# We can also run a command in the environment without activating it first
$ pixi run python -c 'print("Hello World!")'
Hello World!
```

Pixi can also install packages from PyPI. Check Pixi's documentation for a more detailed tutorial on how to use Pixi with Python.

### Installing Command Line Tools (CLI)

CLI tools can be installed inside Pixi environments or globally for the user. The example below installs a tool globally, making it available without activating an environment.

```bash
$ pixi global install jq
└── jq: 1.8.1 (installed)
    └─ exposes: jq

# Confirm the package was installed under the home dir
$ which jq
/home/user/.pixi/bin/jq

# Verify it works
$ jq --version
jq-1.8.1
```

### Finding Packages

Pixi looks for packages in Conda channels. Conda packages are grouped inside channels. By default, Pixi uses the popular conda-forge channel, which hosts over 30k packages.

You can browse available packages in a channel using Pixi's website or directly from the CLI:

```bash
pixi search <package>
```

### Reproducing an Environment

Pixi environments can be reproduced from the `pixi.toml` and `pixi.lock` files. By sharing both of these files, your environment can be reproduced by others. These files are automatically updated when packages are installed with Pixi. Ensure that you keep them up to date under version control.

```bash
# Clone the project files and cd into it
$ git clone <shared-project>
$ cd <shared-project>
$ ls
pixi.lock  pixi.toml  src/

# Install environment
$ pixi install
```

> **Note:** This install command only installs packages managed by Pixi. If you made additional installs with other package managers (e.g. Python's pip or Rust's cargo), you will need separate steps to install the other packages as well.

### Getting Help

While the Conda ecosystem is extensive, some software might not be available and might need to be installed from source. If you run into any issues trying to use a specific software, please reach out using the contacts page. We're happy to help!

---

# Frequently Asked Questions (FAQ)

**Where do I send feedback?**
Please use the feedback form.

**Who can access ORIX?**
ORIX is available to KAUST users only, and access is granted by request and explicit approval. To be eligible, usage must be for internal KAUST research projects, and users must be explicitly whitelisted. In practice, this includes KAUST faculty, students, staff, and postdoctoral researchers working on approved internal projects.

**Is it preferable to run on the PI partition or the freecycle partition?**
It depends on your goal. Use your PI partition when you need predictable progress and guaranteed access to your group's reserved capacity. Use freecycle when you want opportunistic access to idle resources and your workload is preemption-tolerant. In short, use PI resources for reliability and freecycle for extra capacity.

**My PI does not have a reserved partition. What should they do?**
PI partitions were allocated to a select group of faculty identified as the heaviest GPU users on IBEX during the period March 2024 – February 2025. As usage patterns shift over time, new PI partitions may be allocated and existing ones revised. PIs who do not currently have a reserved partition are encouraged to get in touch with Prof. Marco Canini to discuss their context and explore whether a reservation is appropriate. In the meantime, you can still use the shared unreserved resources and the freecycle partition for opportunistic access.

**I work on multiple projects. Am I allowed to use the cluster under a single usage application?**
No. You can work on one project at a time per approved application. If you need to use ORIX for multiple projects, submit a separate application for each project.

**Is there a recommended storage limit to ensure fair usage?**
We do not currently enforce a limit. Storage capacity is initially limited at 200 TB, with a plan to increase by 200 TB increments monthly until 1 PB (after 5 months). Storage is split across two volumes:

- `/home/<uname>`: limited capacity. Use it for your development environment and lightweight files.
- `/mnt/data/u/<uname>`: use this for data-intensive artifacts such as datasets, checkpoints, and outputs.

As a general rule, keep `/home` lean and move heavy project data to `/mnt/data/u/<uname>`.

**How do I decide between regular and preemptible jobs?**
Use regular jobs for work that should not be interrupted. Use preemptible jobs when your workflow can checkpoint and resume, and when you need to scale jobs to using more resources than the limits imposed for regular jobs. If your workload is resilient to interruption, preemptible jobs are a great way to consume idle capacity and gain access to more resources.

**What happens when a preemptible job is interrupted?**
Preemptible jobs can be canceled to make room for higher-priority jobs. On preemption, jobs receive a signal and are granted a 60 second grace period to gracefully stop running tasks. To reduce lost work, design jobs to checkpoint regularly and use Slurm requeue behavior when appropriate.

**Can I install software system-wide for everyone?**
No. Users are expected to manage software in their own home/project environments. We recommend project-level environments so dependencies stay isolated and reproducible.

**What is the best first-run checklist before scaling up?**

1. Confirm your access is approved.
2. Read cluster policy, preemption, and software policy.
3. Run a small validation job first.
4. Add and test checkpointing before launching long runs.
