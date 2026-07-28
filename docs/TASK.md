 Step 1: 

    Read KV Cache Optimization survey paper https://aclanthology.org/2026.findings-acl.1916.pdf (Try to understand the big picture of current KV Cache optimization methods)
    Install vLLM.
    Run any open-source small model locally. (3.2 3B or Qwen 2.5 3B)
    Measure GPU memory usage as context length increase

Step 2:

    Experiment with LMCache. https://github.com/lmcache/lmcache (LMCache Paperhttps://arxiv.org/pdf/2510.09665 )
    Compare GPU memory with and without CPU offloading.

Step 3:

     Reproduce one published method based on your choice. You can find KV cache optimization project in this repository. https://github.com/jjiantong/Awesome-KV-Cache-Optimization 
    Benchmark memory usage, latency, throughput

Step 4:

    And try to think your own contribution based on the previous results.
    One more thing I'd like to emphasize again is that you don't need to invent an entirely new algorithm for your senior project. Your goal is to learn how to conduct research systematically. Even if your project focuses on reproducing and benchmarking existing KV cache optimization techniques on consumer-grade hardware, that is still a worthwhile contribution. A careful empirical evaluation, accompanied by a clear analysis of the trade-offs between memory usage, latency, throughput, and model quality, can make for an excellent senior project.

