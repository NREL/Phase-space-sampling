import socket
import time


def on_cluster():
    hostname = socket.gethostname()
    return False if hostname == "mhassana-34732s" else True


def get_timestamp():
    formatted_time = time.strftime("%d-%b-%y||%H:%M:%S")
    return formatted_time


def get_project_root():
    if on_cluster():
        path = "/home/s1638128/deployment/decomposition-flows"
    else:
        path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp"
    return path


def get_log_root():
    if on_cluster():
        path = "/home/s1638128/tmp/decomposition-flows/log"
    else:
        path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp/log"
    return path


def get_data_root():
    if on_cluster():
        path = "/home/s1638128/deployment/decomposition-flows/datasets"
    else:
        path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp"
    return path


def get_checkpoint_root(from_cluster=False):
    if on_cluster():
        path = "/home/s1638128/tmp/decomposition-flows/checkpoints"
    else:
        if from_cluster:
            path = "/home/conor/Dropbox/phd/projects/decomposition-flows/checkpoints/cluster"
        else:
            path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp/checkpoints"
    return path


def get_output_root():
    if on_cluster():
        path = "/home/s1638128/tmp/decomposition-flows/out"
    else:
        path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp/out"
    return path


def get_final_root():
    if on_cluster():
        path = "/home/s1638128/deployment/decomposition-flows/final"
    else:
        path = "/Users/mhassana/Desktop/BitBucket/OutOfDistribution/GNN/NeuralSplineFlows/nsf-master/experiments/malikExp/final"
    return path


def main():
    print(get_timestamp())


if __name__ == "__main__":
    main()
