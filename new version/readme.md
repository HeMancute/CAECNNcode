#  new version
a new version for network training process.  
trainning with tfrecord and validation process.

# merge all 
The tf.merge_all_summaries() function is convenient, but also somewhat dangerous: it merges all summaries in the default graph, which includes any summaries from previous—apparently unconnected—invocations of code that also added summary nodes to the default graph. If old summary nodes depend on an old placeholder, you will get errors like the one you have shown in your question (and like previous questions as well).
