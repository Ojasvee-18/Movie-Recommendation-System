from model_wrapper import ModelWrapper
print("Starting Wrapper Test")
wrapper = ModelWrapper()
print("Wrapper initialized")
recs = wrapper.get_recommendations('svd', 2000)
print(f"SVD Recommendations: {len(recs)}")
metrics = wrapper.get_metrics()
print(f"Metrics: {metrics}")
