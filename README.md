# PointNet meets Self-Attention Graph Pooling: A Synergistic Approach to Point Cloud Classification

## Abstract
 This research project explores point cloud classification using a combination of PointNet and Self-Attention Graph Pooling architectures. Four variations of architectures were implemented and trained to enhance classification accuracy. The first architecture combines Self-Attention Graph Pooling with node centrality features and xyz coordinates. PointNet is employed as a standalone architecture, capturing local structures using xyz features. Fusion approaches were investigated, including feature concatenation and utilizing PointNet features for subsequent Self-Attention Graph Pooling. Extensive experiments were conducted on the ModelNet10 dataset, showcasing the efficiency of the combined architectures. The fusion approaches demonstrated improved classification accuracy compared to individual architectures, while also reducing model size. The integration of node centrality features further enhanced the discriminative power of the model. This research contributes to the advancement of point cloud classification, highlighting the potential of the combined PointNet and Self-Attention Graph Pooling approach in real-world applications such as object recognition and 3D perception in robotics.


## Review 
The task of this research was to explore and enhance the point cloud classification using feature engineering, particularly by incorporating graph-based representations and node centralities. To achieve this, two main traditional approaches, PointNet and SAGPool, were employed as the basis for investigation. The study focused on developing and evaluating two new fusion approaches, namely FeatureConcat and PointNetBasedGraphPooling, which combined the strengths of PointNet and SAGPool architectures.

Feature engineering played a crucial role in improving the representation of point cloud data. By building graphs and utilizing node centralities, the relationships and significance of each point in the point cloud were effectively captured. The inclusion of node centralities, such as betweenness, closeness, Katz, PageRank, eigenvector, harmonic, and load centralities, provided valuable connectivity information, aiding in more accurate and robust classification.

The traditional approaches, PointNet and SAGPool, served as fundamental baselines for the new fusion approaches. PointNet, being a pioneering architecture in point cloud classification, showcased efficiency in handling unordered data. SAGPool, on the other hand, demonstrated the capability of adaptive feature pooling using graph structures, proving beneficial in capturing global context.

The newly proposed approaches, FeatureConcat and PointNetBasedGraphPooling, exhibited significant improvements in point cloud classification. FeatureConcat effectively combined high-level features from PointNet and SAGPool, leading to remarkable accuracy gains. PointNetBasedGraphPooling leveraged the intersection of PointNet and SAGPool features, further enhancing classification performance.

The results indicated that the fusion of SAGPool and PointNet significantly improved test accuracy compared to the individual approaches. FeatureConcat achieved an impressive test accuracy of 92.07%, while PointNetBasedGraphPooling closely followed with 89.87%. These findings demonstrated the value of integrating graph-based pooling and centralities with PointNet's feature extraction capabilities, resulting in more informative and robust representations for point cloud classification.

In conclusion, this research successfully explored and enhanced point cloud classification through feature engineering. The incorporation of graph-based representations and node centralities proved instrumental in achieving more accurate and efficient classification. The newly proposed fusion approaches, FeatureConcat and PointNetBasedGraphPooling, demonstrated remarkable performance gains, showcasing their potential in advancing point cloud classification tasks. This study contributes valuable insights and practical approaches for handling complex point cloud data and lays the foundation for further research in this field.


## Future works
1)	Diverse Dataset Exploration: To further validate the robustness and generalization of the proposed fusion approaches, it is essential to explore and evaluate their performance on different datasets with varying complexities and sizes. Investigating datasets that encompass a wider range of 3D objects and scenes will provide deeper insights into the approaches' adaptability and effectiveness in real-world scenarios.

2)	Extended Training Analysis: In future research, the effect of extended training periods should be explored to understand how longer training durations impact the convergence and overall performance of the fusion approaches. Additionally, investigating the influence of different batch sizes and learning rates on training dynamics and model performance will help in optimizing the training process and achieving higher accuracy.

3)	Fine-tuning of Fusion Approach Components: The proposed fusion approaches, FeatureConcat and PointNetBasedGraphPooling, utilize MLP and 1D convolution layers, dropout, and batch normalization. Fine-tuning these components and hyperparameters can further enhance the network's performance and efficiency. Optimizing these details will lead to improved feature extraction and pooling, contributing to better classification results.

4)	Ablation Study of Used Features: An ablation study, focusing on the impact of the individual features used in the fusion approaches, would shed light on their individual importance and contribution to the classification performance. Understanding the significance of each feature can help in designing more effective and efficient feature representations for point cloud classification tasks.

Furthermore, we extend an invitation to fellow researchers to join efforts in preparing and submitting this research to a reputable academic conference or journal. By collaborating and sharing insights, we can collectively advance the field of point cloud classification and foster meaningful discussions within the scientific community.


## Acknowledgement
The successful completion of this research and investigation is acknowledged as Mohsen Ebadpour's final project in the 3D Vision course during his Master of Science degree at Amirkabir University of Technology. Gratitude is extended to Dr. Javanmardi, the course lecturer, for providing invaluable guidance and mentorship, which played a crucial role in shaping the direction and outcomes of this academic journey. His support and expertise were highly appreciated throughout the study, serving as a guiding force in the research process.

## License
This project is licensed under the GPL-3.0 License.

## Feedback
If you have any feedback or suggestions for improving this research, please feel free to open an issue in the repository as well as send an email to Mohsen Ebadpour (<m.ebadpour@aut.ac.ir> , <mohsenebadpour@outlook.com>).
