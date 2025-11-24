import pandas as pd
import matplotlib.pyplot as plt

from river.datasets import synth


class DriftGenerator:
    def __init__(self, size_of_dataset, **kwargs):
        self.size_of_dataset = size_of_dataset
        self.params = kwargs
        self.data = None

    def get_data(self, attr_name):
        return getattr(self, attr_name)

    def generate(self):
        pass

    def write_to_csv(self, filename):
        self.data.to_csv(filename, index=False)


class SEADriftDatasetGenerator(DriftGenerator):
    def __init__(self, size_of_dataset, **kwargs):
        super().__init__(size_of_dataset, **kwargs)
        self.variants = []

    def generate(self):
        self.data = pd.DataFrame(columns=[f"feature_{i}" for i in range(3)] + ["target"])

        int = self.size_of_dataset // len(self.variants)
        for variant in self.variants:
            stream_data = self.generate_single_stream(variant, int, self.params)
            self.data = pd.concat([self.data, stream_data], ignore_index=True)

        return self.data

    def generate_single_stream(self, variant, size, kwargs):
        stream = synth.SEA(variant=variant, **kwargs)
        data = pd.DataFrame(columns=[f"feature_{i}" for i in range(3)] + ["target"])
        for x, label in stream.take(size):
            row = x[0], x[1], x[2], label
            data.loc[len(data)] = row
        return data

    def plot_streams(self, size, kwargs):
        plt.figure(figsize=(15, 10))
        for i, variant in enumerate(self.variants):
            stream_data = self.generate_single_stream(variant, size, kwargs)
            plt.subplot(2, 2, i + 1)
            plt.scatter(
                stream_data["feature_0"],
                stream_data["feature_1"],
                c=stream_data["target"],
                cmap="viridis",
            )
            plt.title(f"SEA Stream Variant {variant}")
            plt.xlabel("Feature 0")
            plt.ylabel("Feature 1")
        plt.tight_layout()
        plt.show()

    def plot_single_stream_variant(self, variant, size, kwargs):
        stream_data = self.generate_single_stream(variant, size, kwargs)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            stream_data["feature_0"],
            stream_data["feature_1"],
            c=stream_data["target"],
            cmap="viridis",
        )

        plt.title(f"SEA Stream Variant {variant}")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.legend()
        plt.show()


class DriftPlotter:
    def plot_data(self, data):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            data["feature_0"],
            data["feature_1"],
            c=data["target"],
            cmap="viridis",
        )

        plt.title("SEA Drift Dataset")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.legend()
        plt.show()


class SEADriftDatasetGenerator1(SEADriftDatasetGenerator):
    def __init__(self, size_of_dataset, **kwargs):
        super().__init__(size_of_dataset, **kwargs)
        self.variants = [0, 1, 2, 3]


class HyperplaneDriftDatasetGenerator(DriftGenerator):
    pass


class RBFDatasetGenerator(DriftGenerator):
    pass


class DBDatasetGenerator(DriftGenerator):
    pass


if __name__ == "__main__":
    generator = SEADriftDatasetGenerator1(size_of_dataset=10000)
    # generator.generate()
    # generator.plot_data()
    generator.plot_streams(size=10000, kwargs={})

# TODO:
# 1. Implement HyperplaneDriftDatasetGenerator, RBFDatasetGenerator, and DBDatasetGenerator classes.
# 2. Add unit tests for each generator class.
# 3. Enhance plotting functions with more customization options.
# 4. Optimize data generation for larger datasets.
# 5. Add documentation and usage examples for each class and method.
# 6. Correct kwargs
# 7. Add legend to plots
