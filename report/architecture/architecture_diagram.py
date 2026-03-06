"""Generate AWS architecture diagrams using the `diagrams` library.

Requirements:
    pip install diagrams
    brew install graphviz  # or apt-get install graphviz

Usage:
    python docs/architecture_diagram.py

Produces PNG files in docs/ that the README references.
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECR, ECS, Fargate, Lambda
from diagrams.aws.database import DynamodbTable
from diagrams.aws.integration import SNS, StepFunctions
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import SagemakerTrainingJob
from diagrams.aws.network import ALB
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import SimpleStorageServiceS3 as S3
from diagrams.onprem.ci import GithubActions

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.5",
    "nodesep": "0.8",
    "ranksep": "1.0",
}

# ---------------------------------------------------------------------------
# 1. High-Level Architecture
# ---------------------------------------------------------------------------
with Diagram(
    "SpanishGas MLOps Architecture",
    filename="docs/architecture_high_level",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    outformat="png",
):
    # -- Ingestion --
    with Cluster("Ingestion"):
        s3_raw = S3("Raw Data\n(S3 Upload)")
        lm = Lambda("Pipeline\nTrigger")
        ddb = DynamodbTable("Manifest\n(DynamoDB)")
        s3_raw >> lm
        lm >> Edge(label="idempotent") >> ddb

    # -- Orchestration --
    with Cluster("Pipeline  (Step Functions)"):
        sfn = StepFunctions("Orchestrator")
        with Cluster("SageMaker Processing Jobs"):
            bronze = SagemakerTrainingJob("Bronze")
            silver = SagemakerTrainingJob("Silver")
            gold = SagemakerTrainingJob("Gold")
            train = SagemakerTrainingJob("Train")
            evaluate = SagemakerTrainingJob("Evaluate")
            score = SagemakerTrainingJob("Score")
            drift = SagemakerTrainingJob("Drift")
        sfn >> bronze >> silver >> gold >> train >> evaluate >> score >> drift

    lm >> sfn

    # -- Storage --
    s3_lake = S3("Data Lake\n(S3)")
    sm_reg = SagemakerTrainingJob("Model\nRegistry")

    gold >> Edge(style="dashed") >> s3_lake
    score >> Edge(style="dashed") >> s3_lake
    evaluate >> Edge(label="PR-AUC ≥ 0.70") >> sm_reg

    # -- Monitoring --
    with Cluster("Monitoring"):
        sns = SNS("SNS Alerts")
        cw = Cloudwatch("CloudWatch\nAlarms")
    drift >> sns
    drift >> cw

    # -- Serving --
    with Cluster("Serving"):
        alb = ALB("ALB :80")
        ecs = Fargate("ECS Fargate")
        # Streamlit is inside ECS
    s3_lake >> ecs
    alb >> ecs

    # -- CI/CD --
    with Cluster("CI/CD"):
        gha = GithubActions("GitHub\nActions")
        ecr = ECR("ECR\n(3 images)")
        iam = IAMRole("OIDC\nDeploy Role")
    gha >> ecr
    gha >> iam
    ecr >> Edge(style="dashed") >> lm
    ecr >> Edge(style="dashed") >> ecs


# ---------------------------------------------------------------------------
# 2. Data Pipeline (Medallion Architecture)
# ---------------------------------------------------------------------------
with Diagram(
    "Data Pipeline — Medallion Architecture",
    filename="docs/architecture_data_pipeline",
    show=False,
    direction="LR",
    graph_attr={**graph_attr, "nodesep": "1.0", "ranksep": "1.5"},
    outformat="png",
):
    with Cluster("Raw (7 files)"):
        raw = S3("CSV / JSON\nParquet")

    with Cluster("Bronze"):
        bc = SagemakerTrainingJob("bronze\ncustomer")
        bcm = SagemakerTrainingJob("bronze\ncustomer_month")

    with Cluster("Silver"):
        sc = SagemakerTrainingJob("silver\ncustomer")
        scm = SagemakerTrainingJob("silver\ncustomer_month")

    with Cluster("Gold"):
        gm = SagemakerTrainingJob("gold_master\n56 features")

    raw >> bc
    raw >> bcm
    bc >> sc
    bcm >> scm
    sc >> gm
    scm >> gm


# ---------------------------------------------------------------------------
# 3. ML Training & Evaluation
# ---------------------------------------------------------------------------
with Diagram(
    "ML Training & Evaluation",
    filename="docs/architecture_ml_training",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    outformat="png",
):
    gm = S3("Gold Master")
    train = SagemakerTrainingJob("XGBoost\nTraining")
    evaluate = SagemakerTrainingJob("Evaluate")
    gm >> train >> evaluate

    s3_models = S3("S3 models/")
    sm_reg = SagemakerTrainingJob("SM Registry")

    evaluate >> Edge(label="PR-AUC ≥ 0.70", color="green") >> s3_models
    evaluate >> Edge(label="PR-AUC ≥ 0.70", color="green") >> sm_reg
    evaluate >> Edge(label="PR-AUC < 0.70", color="red", style="dashed") >> SagemakerTrainingJob("Rejected")


# ---------------------------------------------------------------------------
# 4. Serving & Dashboard
# ---------------------------------------------------------------------------
with Diagram(
    "Serving & Dashboard",
    filename="docs/architecture_serving",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    outformat="png",
):
    alb = ALB("ALB :80")
    ecs = Fargate("ECS Fargate\nStreamlit")
    s3_scored = S3("S3 scored/\neval / drift / reco")
    ddb = DynamodbTable("DynamoDB\nmanifest")

    alb >> ecs
    s3_scored >> ecs
    ddb >> ecs


# ---------------------------------------------------------------------------
# 5. CI/CD & Deployment
# ---------------------------------------------------------------------------
with Diagram(
    "CI/CD & Deployment",
    filename="docs/architecture_cicd",
    show=False,
    direction="LR",
    graph_attr={**graph_attr, "ranksep": "1.2"},
    outformat="png",
):
    gha = GithubActions("GitHub Actions")
    iam = IAMRole("OIDC\nDeploy Role")

    with Cluster("Deploy"):
        ecr = ECR("ECR\n(3 images)")
        lm = Lambda("Lambda")
        ecs = Fargate("ECS Fargate")

    with Cluster("Retrain"):
        sfn = StepFunctions("Step Functions")

    gha >> iam
    gha >> ecr
    ecr >> lm
    ecr >> ecs
    gha >> sfn


print("✓ All diagrams generated in docs/")
