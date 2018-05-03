#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/tensor_format.h"
#include <iostream>
#include "louvain.h"

using std::vector;
using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

typedef vector<int> Cluster;

struct Merger {
	// It is called when the algorithm merge the nodes into the cluster.
	Cluster operator()(std::vector<louvain::Node<Cluster> > const& nodes, std::vector<int> idxs) const{
		// Select the most popular person
		Cluster c;
        int s = 0;
        for(auto idx : idxs){
           s += nodes[idx].payload().size();
		}
        c.reserve(s);
        for(auto idx : idxs){
            Cluster* ci = (Cluster*) &(nodes[idx].payload());
            c.insert(c.end(), ci->begin(), ci->end());
		}
		return c;
	}
};

REGISTER_OP("Louvain")
    .Input("adj: bool")
    .Input("weights: int32")
    .Input("n_clusters: int32")
    .Output("supernode_assign: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {        
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class LouvainOp : public OpKernel {
 public:
      explicit LouvainOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& adj = context->input(0);
    auto adj_arr = adj.tensor<bool, 3>();
    const Tensor& weights = context->input(1);
    auto weights_arr = weights.tensor<int, 3>();
    const Tensor& n_clusters = context->input(2);
    auto n_clusters_arr = n_clusters.tensor<int, 1>();
    // Create an output tensor
    Tensor* supernode_assign = NULL;
      
    const TensorShape& adj_shape = adj.shape();
    TensorShape out_shape = adj_shape;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &supernode_assign));
    
    auto output_shape = out_shape.dim_sizes();
    auto output_flat = supernode_assign->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = output_flat.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    auto supernode_assign_arr = supernode_assign->tensor<float, 3>();

    const int batch_size = output_shape[0];
    std::function<void(int64, int64)> shard;
    shard = [&adj_arr, &weights_arr, &n_clusters_arr, &supernode_assign_arr, &output_shape](int64 start, int64 limit) {
        for (int graph = start; graph < limit; ++graph) {
            int n_clus = n_clusters_arr(graph);
            std::vector<int> ids;
            for (int m = 0; m < output_shape[1]; m++) {
                if (adj_arr(graph, m, m)) {
                    ids.push_back(m);
                }
            }
			if (ids.size() > 0) {
				if (n_clus > 1 && n_clus < ids.size()) {
					int total_links = 0;
					std::vector<louvain::Node<Cluster>> vertices(ids.size());
					for (int m = 0; m < ids.size(); m++) {
						vertices[m].payload().push_back(ids[m]);
						for (int n = 0; n < ids.size(); n++) {
							if (m != n && adj_arr(graph, ids[m], ids[n])) {
								vertices[m].neighbors().push_back(std::pair<int,int>(n, weights_arr(graph, ids[m], ids[n])));
								total_links += weights_arr(graph, ids[m], ids[n]);
							}
						}
					}
					louvain::Graph<Cluster, Merger> g(total_links, std::move(vertices));

					do {
						const size_t nedges = g.edges();
						const size_t nnodes = g.nodes().size();
						g = g.nextLevel();
						if (g.edges() == nedges && g.nodes().size() == nnodes) {
							break;
						}
					} while(g.nodes().size() > n_clus);

					int count = 0;
					for (auto cluster : g.nodes()) {
						for (auto node : cluster.payload()) {
							supernode_assign_arr(graph, node, count) = 1.;
						}
						count++;
					}
				}
				else {
					for(auto i : ids) {
						supernode_assign_arr(graph, i, i) = 1.;
					}
				}
			}
        }
    };

    // This is just a very crude approximation
    const int64 single_cost = 10000 * output_shape[1] * output_shape[2];

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, single_cost, shard);
  }

 private:
    TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("Louvain").Device(DEVICE_CPU), LouvainOp);
