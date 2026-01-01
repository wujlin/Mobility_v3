# UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces

Yuanshao Zhu $^{1,2,3,*}$ , James Jianqiao Yu $^{4,\dagger}$ , Xiangyu Zhao $^{2,\dagger}$ , Xun Zhou $^{4}$ , Liang Han $^{4}$ , Xuetao Wei $^{1}$ , Yuxuan Liang $^{3,\dagger}$

$^{1}$  Southern University of Science and Technology,  $^{2}$  City University of Hong Kong

<sup>3</sup> The Hong Kong University of Science and Technology (Guangzhou)

$^{4}$  Harbin Institute of Technology, Shenzhen

yuanshao@ieee.org, jqyu@ieee.org, xianzhao@cityu.edu.hk

zhouxun2023@hit.edu.cn, han.liang@hit.edu.cn

weixt@sustech.edu.cn, yuxliang@outlook.com

# Abstract

Building a universal trajectory foundation model is a promising solution to address the limitations of existing trajectory modeling approaches, such as task specificity, regional dependency, and data sensitivity. Despite its potential, data preparation, pre-training strategy development, and architectural design present significant challenges in constructing this model. Therefore, we introduce UniTraj, a Universal Trajectory foundation model that aims to address these limitations through three key innovations. First, we construct WorldTrace, an unprecedented dataset of 2.45 million trajectories with billions of GPS points spanning 70 countries, providing the diverse geographic coverage essential for region-independent modeling. Second, we develop novel pre-training strategies—Adaptive Trajectory Resampling and Self-supervised Trajectory Masking—that enable robust learning from heterogeneous trajectory data with varying sampling rates and quality. Finally, we tailor a flexible model architecture to accommodate a variety of trajectory tasks, effectively capturing complex movement patterns to support broad applicability. Extensive experiments across multiple tasks and real-world datasets demonstrate that UniTraj consistently outperforms existing methods, exhibiting superior scalability, adaptability, and generalization, with WorldTrace serving as an ideal yet non-exclusive training resource. The implementation codes and full dataset are available in the https://github.com/Yasoz/UniTraj.

# 1 Introduction

Trajectory data, as the digital footprints of human movement, is becoming a fundamental data source for understanding mobility patterns and transforming urban intelligence [3]. These spatiotemporal sequences unlock critical insights across diverse applications: from optimizing transportation networks that alleviate congestion in megacities, enhancing location-based services that personalize user experiences [18, 2], to powering logistics systems that determine the efficiency of global supply chains [12, 24, 36]. Despite their significance, extracting meaningful patterns (from statistical methods to deep learning [25]) of trajectory data presents profound challenges due to their inherent complexity, varying lengths, irregular sampling rates, and region-specific characteristics.

As trajectory data continues to expand exponentially, three critical limitations in current approaches have become increasingly apparent: (1) Task Specificity: Current approaches are typically designed

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/799f21b8c851d684b4cc0abb2d2ce6ce5d3f314358de2e1e4393258c42b80ed6.jpg)



Figure 1: Overview of this work, we propose a trajectory foundation model and also collect a worldwide trajectory dataset. The pre-trained UniTraj can be used as a backbone while adapters are trained for different regions and tasks.


for single-purpose applications, limiting their generalizability and requiring substantial re-engineering for new tasks. (2) Regional Dependency: Many models are developed and trained on data from specific geographic regions, making them ineffective when applied to different locations with distinct mobility patterns and infrastructure. (3) Data Sensitivity: Real-world trajectory data often contains noise, irregular sampling, or missing entries, making models highly sensitive to data quality and necessitating extensive preprocessing, which reduces robustness. These limitations point to a fundamental gap: the absence of a universal foundation model capable of operating across diverse tasks, geographic regions, and data quality levels. While foundation models have revolutionized NLP [1, 7] and CV [9, 13] by providing versatile, pre-trained architectures that generalize across domains, trajectory analysis has not yet benefited from this paradigm shift. Creating this model would transform trajectory intelligence from its current fragmented state to a unified approach with significantly enhanced generalization capabilities [27, 39].

However, building such a model presents key challenges: (1) Data preparation: The first challenge is to prepare a sufficiently diverse trajectory dataset that spans different geographic regions and appropriate sampling rates. Existing datasets lack sufficient geographic diversity and scale, also limited by proprietary restrictions and collection costs. This data scarcity severely hampers model generalizability and cross-regional research efforts on a global scale. (2) Pre-training Strategy: Developing robust and scalable pre-training strategies is another challenge. Real-world trajectory data exhibits heterogeneous quality with noising, varying sampling rates, and missing points. Effective pre-training must accommodate these inconsistencies while learning robust representations that transfer across diverse contexts. (3) Model Design: The last challenge involves selecting and tailoring an effective model architecture. A universal foundation model requires an architecture that balances adaptability across tasks with computational efficiency, capturing complex spatio-temporal dependencies without overfitting to specific regional information or trajectory patterns.

To address these challenges, we introduce Universal Trajectory foundation model (UniTraj) supported by three key innovations. As shown in Figure 1, we firstly construct WorldTrace, the first trajectory dataset with large-scale, high-quality, and global distribution, which provides the essential foundation for region-agnostic modeling. Then, we design several novel pre-training strategies—adaptive resampling and self-supervised masking—that enable robust learning from heterogeneous trajectory data with varying sampling rates and quality, bridging the gap between regional variations and inconsistent data. Finally, we design a flexible model architecture that captures complex spatio-temporal dependencies while adapting to diverse trajectory tasks, creating a versatile backbone for trajectory modeling. Collectively, UniTraj achieves task-adaptive, region-independent, and data quality resilience, delivering a scalable and efficient solution for trajectory analysis applications. In summary, our research makes the following key contributions:

- We introduce WorldTrace, a pioneering trajectory dataset spanning 70 countries with 2.45 million trajectories and billions of GPS points. Its unprecedented global diversity and quality overcome the limitations of existing region-specific datasets, offering a comprehensive and open groundwork for facilitating trajectory modeling research.

- We propose UniTraj, trained on WorldTrace and equipped with novel pre-training and masking strategies that effectively capture complex spatio-temporal dependencies. This model significantly

enhances generalizability across tasks and geographical contexts, adapts to the heterogeneity of data, and provides a scalable and efficient solution for a wide range of trajectory analysis applications.

- We demonstrated the effectiveness of UniTraj through comprehensive experiments on multiple trajectory analysis tasks. The results show significantly improved performance of zero-shot and fine-tuning settings, confirming its potential as a versatile backbone for diverse trajectory modeling tasks, performing optimally when trained on diverse and high-quality datasets like WorldTrace.

# 2 Related Work

Trajectory Datasets. Trajectory datasets are foundational for advancing mobility research, yet existing collections vary (geographic coverage, data quality, and granularity) considerably in their utility and limitations. Well-known datasets, such as GeoLife [46], collected over five years by 182 users, has contributed significantly to fields like travel mode detection [5] and traffic flow analysis [19]. However, its limited geographic coverage and participant diversity restrict its generalizability. Ehicle-focused datasets such as Porto [28], T-drive [42], and Electric Vehicle Data [34] provide valuable mobility insights but frequently exhibit low or inconsistent sampling rates that complicate analysis. Synthetic alternatives like SynMob [49] offer uniform sampling but lack the regional diversity and quality variations essential for robust model development. Proprietary collections including GAIA [8] and Grab-Posisi [15] contain high-quality data but remain largely inaccessible due to regulatory and commercial constraints. These limitations—geographic constraints, sampling irregularities, and access restrictions—collectively impede the development of universal trajectory models. The community urgently needs comprehensive, openly accessible datasets with global coverage to advance trajectory modeling research and enable effective model generalization.

Foundation Models. The success of foundation models in natural language processing and computer vision, exemplified by BERT [7], GPT-3 [1], and Vision Transformers [9], has demonstrated how large-scale pretraining can yield highly generalizable representations across diverse tasks. This paradigm has recently extended to time series and spatio-temporal domains, with models like TST [44], TimeFM [6], and Moirai [38] leveraging Transformer architectures to capture temporal dependencies. In spatio-temporal prediction specifically, approaches such as UniST [43], Opencity [20], and ClimaX [29] have shown promise in traffic flow and climate modeling, respectively. However, these models often remain tailored to specific tasks or regions, limiting their broader applicability. Trajectory-specific models like TrajGDM [4], BigCity [41], and TrajFM [23] address certain tasks but lack the scalability and robustness needed for cross-task or cross-region applications. While unsupervised learning approaches like MAE [13] and TimeFM [6] have proven effective for images and time series, trajectory modeling presents unique challenges that demand greater flexibility to accommodate diverse mobility patterns, geographic contexts, and sampling characteristics without extensive task-specific modifications. To summary, there remains a pressing need for trajectory foundation models that unify multiple tasks within a single framework, providing robust, transferable representations that generalize across tasks and handle data variability while maintaining computational efficiency.

# 3 Preliminary

Definition 1: (Trajectory). A trajectory represents the sequential record of movement through space over time. Formally, a trajectory  $\pmb{\tau}$  of length  $n$  is expressed as a sequence of continuously sampled GPS points:  $\pmb{\tau} = \{p_1, p_2, \dots, p_n\}$ , where each point  $p_i = \langle \mathrm{lng}_i, \mathrm{lat}_i, t_i \rangle$  denotes the spatial coordinates (longitude and latitude) at timestamp  $t_i$ . The sampling interval between consecutive points is defined as  $\Delta t_i = t_i - t_{i-1}$ , for  $i = 2, \dots, n$ . These intervals may be uniform within or across trajectories, or vary significantly based on data collection methods and environmental factors.

Definition 2: (Trajectory Dataset). A trajectory dataset comprises multiple trajectories, each capturing the movement of an object over time. Formally, it is given by  $\mathcal{D} = \{\tau_1,\tau_2,\dots ,\tau_{|\mathcal{D}|}\}$ , where  $|\mathcal{D}|$  denotes the total number of trajectories in the dataset. These collections may vary in geographic coverage, sampling rates, and quality depending on their source and application scenario.

Problem Statement: (Universal Trajectory Modeling). Building upon the above definitions, this study aims to develop a universal foundation model for trajectory data that can adapt to diverse tasks and geographic contexts while accommodating heterogeneous data sources. Formally, consider a set

of trajectories  $\mathcal{D} = \{\pmb{\tau}_i\}_{i=1}^{|\mathcal{D}|}$ , where each  $\pmb{\tau}_i$  is defined as in Definition 1. The goal is:

$$
F: \boldsymbol {\tau} \mapsto \mathbf {h} \in \mathbb {R} ^ {d}, \tag {1}
$$

which projects a raw trajectory  $\tau$  into a d-dimensional representation  $\mathbf{h}$ . This function  $F(\cdot)$  must capture intrinsic spatio-temporal patterns within trajectories while demonstrating three key capabilities: (1) task adaptability across various applications including classification, prediction, and anomaly detection; (2) region independence, enabling zero-shot generalization to different geographic contexts; and (3) resilience to data quality variations, effectively handling inconsistent sampling rates, varying trajectory lengths, and noise without extensive preprocessing or task-specific re-engineering.

# 4 Methodology

In this section, we describe the methodology for developing UniTraj, addressing the key challenges outlined in the introduction. Our approach is structured around answering three fundamental questions: (1) How to construct a diverse and high-quality trajectory dataset that enables cross-regional generalization? (2) How to develop robust and scalable pre-training strategies that accommodate heterogeneous trajectory data? and (3) How to design an effective model architecture that adapts across diverse trajectory tasks?

# 4.1 WorldTrace Dataset Construction

To address the data preparation challenge, we introduce WorldTrace, a large-scale, globally distributed trajectory dataset specifically designed to support universal trajectory modeling. Below, we introduce our data acquisition process, preprocessing pipeline, and key dataset statistics, demonstrating WorldTrace's suitability as a foundation for developing robust and generalizable trajectory foundation models. Detailed information on processing, analysis, and copyright can be found in Appendix A. The full dataset is available on the Hugging Face<sup>3</sup> and ModelScope<sup>4</sup> platforms.

Data Acquisition. We sourced raw trajectory data from OpenStreetMap (OSM) GPS traces [30], focusing on contributions uploaded between 2021-2023 and tagged for motorized movement to ensure data currency and relevance. This approach minimizes device heterogeneity and outdated data impacts. All collected data is stored in the standardized GPX format (an XML schema), containing latitude, longitude, timestamps, and optional metadata, providing a uniform structure that simplifies parsing and preprocessing. During acquisition, we implemented preliminary filtering to exclude trajectories with obvious anomalies such as coordinates outside valid ranges or duplicate entries.


Table 1: Summary statistics of WorldTrace.


<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of Trajectories</td><td>2.45 Million</td></tr><tr><td>Total Raw Points</td><td>8.8 Billion</td></tr><tr><td>Geographical Covered</td><td>70 Countries</td></tr><tr><td>Sampling Interval</td><td>1 sec (normalized)</td></tr><tr><td>Time Span</td><td>08/2021 – 12/2023</td></tr><tr><td>Avg. Duration</td><td>6 min</td></tr><tr><td>Avg. Distance</td><td>5.73 km</td></tr><tr><td>Avg. Speed</td><td>48.0 km/h</td></tr></table>

Data Preprocessing. Our preprocessing pipeline balances preserving authentic movement patterns with removing noise and inconsistencies, which includes the following steps:

1. Normalization: The original data had a high sampling frequency of up to  $10\mathrm{Hz}$ , causing redundancy and increased storage demands. We therefore resampled trajectories to a uniform rate of one point per second (1 Hz), preserving essential motion details while reducing data size. In addition, by standardizing trajectories to 1s/point, we can perform better resampling during subsequent model training to accommodate frequency inconsistent issues.

2. Filtering: We discarded trajectories with fewer than 32 points or covering distances below 100 meters, as such short trips often lack meaningful patterns and introduce noise. Following established practices [5], we also removed trajectories containing implausible speeds (e.g., exceeding  $120\mathrm{km / h}$ ), typically caused by GPS errors or anomalies. We also apply distance- and loop-based outlier detection to identify and remove trajectories that deviate markedly from the expected path.

3. Calibration: Given that GPS signals can suffer from errors due to building obstructions, multipath effects, and receiver noise [14], we applied map-matching techniques [40] to align raw GPS points with underlying road networks. This calibration step is common practice in trajectory data processing and is widely used in data collection and related research to correct positioning errors [8, 37], improve spatial accuracy, and make trajectory analysis more reliable.

Data Analysis and Statistics. After acquiring and preprocessing the raw trajectory data, we conducted an in-depth analysis to examine the characteristics and quality of the WorldTrace dataset. Table 1 summarizes key statistics of WorldTrace. Overall, the dataset contains approximately 2.45 million trajectories and 8.8 billion raw GPS points, covering 70 countries across all inhabited continents. The data spans August 2021 to December 2023, with an average trajectory duration of about six minutes (with normalized to a 1-second sampling interval), an average distance of  $5.73\mathrm{km}$  and an average speed of  $48.0\mathrm{km/h}$ . The number of points per trajectory ranges from 32 to more than 600, averaging around 358 points. Collectively, these attributes confirm WorldTrace's suitability for developing universal trajectory models that can address varied spatiotemporal patterns and broad geographical contexts.

# 4.2 Pre-Training Strategies

Having established a diverse trajectory dataset, we develop robust pre-training strategies to learn robust and transferable spatio-temporal representations. Rather than relying on task-specific supervision, we leverage unannotated trajectory data to capture both local and global movement patterns. To address the heterogeneous data quality challenges (varying sampling rates, differing lengths, and missing points) posed by real-world trajectory, we propose two strategies tailored specifically for trajectory: Adaptive Trajectory Resampling and Self-supervised Trajectory Masking. Due to space limitations, more details and analysis about pre-training strategies can be found in Appendix B.

Adaptive Trajectory Resampling (ATR). Real-world trajectory data often exhibits inconsistent sampling intervals and lengths due to diverse collection standards, device capabilities, and user behaviors. Such discrepancies challenge model generalization, as features learned under one sampling regime may not transfer to another. Inspired by common practice of multi-scale representation learning, ATR strategy addresses these issues through two complementary components:

- Dynamic Multi-Scale Resampling. This approach dynamically adjusts sampling frequency based on trajectory length, ensuring shorter trajectories retain fine-grained detail while longer ones are efficiently compressed. Specifically, we design a logarithmic resampling function  $R(n)$  to implement this strategy:

$$
R (n) = R _ {\min } + \left(1 - R _ {\min }\right) \cdot \frac {\ln \left(n - n _ {\min } + 1\right)}{\ln \left(n _ {\max } - n _ {\min } + 1\right)}, \tag {2}
$$

where  $n_{\mathrm{min}}$  and  $n_{\mathrm{max}}$  define thresholds for trajectory lengths considered "short" or "long", and  $R_{\mathrm{min}}$  is the minimum sampling ratio. This logarithmic function creates a smooth transition in sampling density ( $n_{\mathrm{min}} < n < n_{\mathrm{max}}$ ), providing three key benefits: (1) preserving critical motion patterns across trajectory lengths, (2) reducing overfitting by limiting redundancy in densely sampled data, and (3) exposing the model to diverse temporal resolutions during training.

- Interval Consistent Resampling. This component focuses on the sampling rate, imposing a uniform time interval  $\Delta t$  between consecutive points within each track:

$$
\boldsymbol {\tau} ^ {\prime} = \left\{p _ {k _ {j}} \mid k _ {j} = 1 + (j - 1) \Delta t, j = 1, 2, \dots , m \right\}. \tag {3}
$$

By ensuring consistent spacing, this approach simplifies downstream modeling by creating regular temporal structures that make time-dependent patterns easier to learn, while mitigating complications from missing data or irregular sampling.

Combining these approaches, ATR enables models to learn representations that generalize across varying sampling rates and trajectory lengths (analysis presented in Appendix B.1), which is a critical capability for universal trajectory modeling.

Self-supervised Trajectory Masking (STM). Trajectory data is often incomplete or irregular due to device limitations, communication failures, and environmental factors. Motivated by masked auto-encoding methods from visual and language models, we introduce a tailored self-supervised

trajectory masking strategy, in which part of the input trajectory is hidden, forcing the model to infer local and global dependencies. Given a resampled trajectory  $\tau' = \{p_1, p_2, \dots, p_n\}$ , we define a masking function  $\mathcal{M}(\tau', r)$  that replaces a fraction  $r$  of points with a [MASK] tokens:

$$
\tilde {\tau} = \mathcal {M} \left(\tau^ {\prime}, r\right) = \left\{p _ {1}, \dots , \left[ \mathrm {M A S K} \right] _ {i \in \mathbf {I}}, \dots , p _ {n} \right\}, \tag {4}
$$

where  $\mathbf{I} \subseteq \{1, 2, \dots, n\}$  and  $r = |\mathbf{I}| / n$ . To comprehensively address different data incompleteness scenarios, (see Appendix B.2 for details) we employ four complementary masking strategies:

- Random Masking: Uniformly samples points to mask  $(\mathbf{I}_{\mathrm{rand}} \sim \mathrm{Uniform}(\{1,2,\dots,n\}))$ , forcing the model to infer both short-range and long-range dependencies. By forcing the reconstruction of randomly omitted points, the approach enhances the model's ability to generalize to diverse gaps.

- Block Masking: Conceals consecutive points  $(\mathbf{I}_{\mathrm{block}} = \{k, k + 1, \dots, k + b - 1\})$  to simulate sensor failures, encouraging reconstruction of continuous segments. This approach prompts the model to utilize surrounding context for reconstructing entire missing segments, encouraging it to capture longer-range dependencies.

- Key Points Masking: Identifies and masks critical turning points using the Ramer-Douglas-Peucker algorithm [10]:  $\mathbf{I}_{\mathrm{key}} = \{p_k \mid d_{\max}(p_k, \overline{p_1 p_n}) > \epsilon\}$  ( $d_{\max}(\cdot)$  is the maximum perpendicular distance between point  $p_k$  and line  $\overline{p_1 p_n}$ ,  $\epsilon$  is the threshold). This focuses learning on structurally significant points (sharp turns or notable speed changes) that define the trajectory's shape.

- Last N Masking: Masks final trajectory points  $(\mathbf{I}_{\mathrm{last}} = \{n - N + 1, n - N + 2, \dots, n\})$ . This setting emulates real-world forecasting tasks where future data is unavailable and must be inferred from historical observations, making it particularly effective for prediction scenarios.

# 4.3 Universal Trajectory Modeling

To effectively leverage the diverse trajectory data and robust pre-training strategies described above, we need to design a model architecture that can capture local and global patterns while freeing itself from regional and task-specific constraints. Our motivation for adopting this structure design is as follows: (1) We need an architecture that can be generalized to a wide range of tasks without extensive restructuring. Therefore, we adopted minimal trajectory data information (latitude, longitude, and timestamp) and ignored other region-bound information such as POI and geographical context. (2) This structure uses the reconstruction of missing points in partial observations as a proxy task and can inherit the masking strategy introduced earlier. (3) The separation of encoding and decoding enables flexible application to various downstream tasks through transfer learning or fine-tuning. More details about the architecture and parameters can be found in Appendix C.

Building Trajectory Embedding. Effective trajectory modeling requires transforming raw spatial and temporal data into structured embeddings that capture both local and global movement patterns. To ensure the generality of the model, we only use the latitude, longitude, and time information of the trajectory, and embed the spatial and temporal components separately to form a unified representation. For the spatial component, we normalize trajectory and map them into a  $d$ -dimensional space using a 1D convolutional, yielding a spatial embedding  $h_i^s$ . Similarly, the temporal component, based on the time intervals  $\Delta t_i$ , is embedded into the same  $d$ -dimensional space via a linear layer, resulting in a temporal embedding  $h_i^t$ . This decoupled design enables the model to effectively learn relative movement and temporal dependencies, and also cope with situations where one component may be absent. Beyond point-wise embedding, modeling the relationships between trajectory points is critical for understanding movement patterns. We adopt Rotary Position Encoding (RoPE) [32], which applies rotational transformations in the embedding space. The advantage of RoPE is its ability to preserve relative positional relationships while allowing for flexible encoding of spatial-temporal patterns across varying trajectory scales.

Adaptive Representation Learning. Based on the trajectory embeddings, we use a encoder-decoder architecture with RoPE-enhanced attention mechanism to adaptively learn a general representation of trajectories. The encoder processes the visible points in a trajectory those that are unmasked during training. Given a masked trajectory  $\tilde{\tau} = \{p_1,\dots ,[\mathrm{MASK}]_{i\in \mathbf{I}},\dots ,p_n\}$ , we first extract the embedding representations of the unmasked points  $\mathbf{H} = \{h_1,h_2,\dots ,h_m\}$  (where  $m\leq n$  and  $i\notin \mathbf{I}$ ) through the embedding steps. The encoder, denoted as  $\mathbf{E}_{\theta}$ , processes these visible embeddings to generate latent representations:  $z_{\mathrm{enc}} = \mathbf{E}_{\theta}\left(\mathbf{H}\right)$ . The decoder reconstructs masked trajectory points based on the latent embeddings produced by the encoder. It receives the visible embeddings and mask


Table 2: Performance comparison of UniTraj with trajectory recovery tasks. The results are reported in MAE and RMSE with meters. Bold denotes the best results and underline denotes the second-best results.


<table><tr><td rowspan="2">Methods</td><td colspan="2">WorldTrace</td><td colspan="2">Chengdu</td><td colspan="2">Xi&#x27;an</td><td colspan="2">GeoLife</td><td colspan="2">Grab-Posisi</td><td colspan="2">Porto</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Linear</td><td>427.68</td><td>516.15</td><td>205.74</td><td>258.52</td><td>176.49</td><td>220.87</td><td>196.85</td><td>249.76</td><td>507.41</td><td>617.28</td><td>396.61</td><td>482.39</td></tr><tr><td>DHTR</td><td>220.35</td><td>302.47</td><td>75.19</td><td>98.68</td><td>62.85</td><td>83.43</td><td>80.04</td><td>168.25</td><td>351.20</td><td>415.16</td><td>194.37</td><td>232.59</td></tr><tr><td>Transformer</td><td>130.82</td><td>147.62</td><td>55.23</td><td>62.85</td><td>45.85</td><td>51.96</td><td>94.68</td><td>113.77</td><td>136.58</td><td>163.29</td><td>104.36</td><td>126.96</td></tr><tr><td>DeepMove</td><td>51.16</td><td>62.29</td><td>29.32</td><td>39.02</td><td>27.31</td><td>35.67</td><td>86.38</td><td>107.78</td><td>126.93</td><td>168.07</td><td>136.66</td><td>174.96</td></tr><tr><td>TrajBERT</td><td>58.13</td><td>70.14</td><td>26.48</td><td>33.83</td><td>19.45</td><td>25.13</td><td>34.53</td><td>43.24</td><td>112.68</td><td>136.24</td><td>78.77</td><td>99.23</td></tr><tr><td>TrajFM</td><td>47.64</td><td>58.92</td><td>19.10</td><td>25.09</td><td>18.86</td><td>24.13</td><td>59.34</td><td>64.24</td><td>107.64</td><td>130.69</td><td>71.15</td><td>92.96</td></tr><tr><td>UniTraj (zero-shot)</td><td>10.22</td><td>13.56</td><td>11.98</td><td>20.94</td><td>8.93</td><td>13.83</td><td>37.21</td><td>63.89</td><td>114.07</td><td>167.01</td><td>78.28</td><td>100.14</td></tr><tr><td>Improvement(%)</td><td>↑78.55</td><td>↑76.99</td><td>↑37.28</td><td>↑16.54</td><td>↑52.65</td><td>↑42.69</td><td>↓7.76</td><td>↓47.46</td><td>↓5.97</td><td>↓27.79</td><td>↓10.02</td><td>↓7.72</td></tr><tr><td>UniTraj (fine-tune)</td><td>6.94</td><td>9.67</td><td>6.92</td><td>10.41</td><td>6.50</td><td>9.93</td><td>23.23</td><td>34.70</td><td>48.95</td><td>69.23</td><td>60.18</td><td>79.76</td></tr><tr><td>Improvement(%)</td><td>↑85.43</td><td>↑83.59</td><td>↑63.77</td><td>↑58.51</td><td>↑65.54</td><td>↑58.85</td><td>↑32.73</td><td>↑19.75</td><td>↑54.52</td><td>↑47.03</td><td>↑15.42</td><td>↑14.20</td></tr></table>

tokens, which are initialized as learnable vectors representing missing positions. The full sequence is created by merging the encoded visible embeddings with the mask tokens, preserving the original structure of the trajectory:

$$
\mathbf {z} _ {\mathrm {d e c}} = \operatorname {R e o r d e r} \left(\left\{ \begin{array}{l l} z _ {i} = z _ {\text {e n c}, j} & \text {i f} i = \operatorname {I n d e x} (j), i \notin \mathbf {I} \\ [ \text {M A S K} ] & \text {i f} i \in \mathbf {I} \end{array} \right\}\right), \tag {5}
$$

where  $z_{\mathrm{enc},j}$  corresponds to the  $j$ -th encoder output. The decoder then processes the reordered sequence to predict the missing trajectory points:  $\hat{\tau} = \mathrm{Linear}(\mathbf{D}_{\phi}(\mathbf{z}_{\mathrm{dec}}))$ . The model is trained to minimize the reconstruction loss between the predicted and original points at the masked positions:

$$
\mathcal {L} = \frac {1}{| \mathbf {I} |} \sum_ {i \in \mathbf {I}} \| f _ {\theta , \phi} (\tilde {\boldsymbol {\tau}}) _ {i} - \boldsymbol {\tau} _ {i} \| ^ {2}, \tag {6}
$$

where  $f_{\theta ,\phi}(\tilde{\tau})$  represents the encoder-decoder network, and  $i$  refers to the masked positions.

# 5 Experiments

# 5.1 Experimental Setups

Datasets. We evaluate UniTraj on six diverse real-world trajectory datasets representing different collection scenarios, quality levels, motion patterns, and geographic regions. These include WorldTrace, Chengdu, Xi'an, GeoLife, Grab-Posisi, and Porto. Detailed summary are provided in Appendix D.1.

# 5.2 Task Applicability Analysis

We explore the applicability and generalizability of UniTraj to various data and downstream tasks, e.g., trajectory recovery, prediction, classification, and generation tasks. Due to space constraints, we provide the detailed setup and the results of generation task in Appendix D.2. It is important to clarify that our work aims to develop a general-purpose trajectory foundation model that generalizes across diverse geographic regions without region-specific dependencies, validating its effectiveness as a backbone supporting real-world trajectory applications across geographical contexts. Existing trajectory representation learning methods inherently rely on region-bound information (POIs, road networks, etc.) [16, 22, 26, 48], which contradicts our initial goal of region-independent modeling. UniTraj extracts meaningful representations solely from trajectory points without requiring auxiliary geographic context. Therefore, we deliberately excluded these methods from our baseline comparison as their architectural dependency on regional knowledge fundamentally diverges from our objective of developing a globally deployable model.

Trajectory Recovery. Table 2 presents a comprehensive comparison of UniTraj against established baselines across six datasets, revealing patterns that illuminate fundamental capabilities in trajectory reconstruction. The performance disparity between UniTraj and previous methods is particularly pronounced in geographically diverse and quality-variable datasets, where it demonstrates substantial resilience to regional variations. In the zero-shot setting, UniTraj achieves remarkable results, confirming it effectively captures transferable spatio-temporal patterns without requiring additional fine-tuning. The performance difference becomes particularly instructive when analyzing low-quality datasets like GeoLife and Grab-Posisi, with their highly irregular sampling intervals and multiple

travel modes. It demonstrates the effectiveness of our adaptive resampling strategy in handling temporal heterogeneity. The Chengdu and Xi'an datasets reveal another critical aspect of UniTraj's capabilities, models trained on high-quality data exhibit reliable transferability and achieve optimal results even in zero-shot scenarios. When fine-tuned, UniTraj achieves the lowest error scores across all datasets, demonstrating UniTraj's superior generalizability across diverse geographic regions. For instance, on GeoLife, UniTraj's fine-tuned performance (MAE 23.23) reduces error by  $32.73\%$  compared to TrajBERT, showcasing its effectiveness with complex travel patterns and lower-quality data. These results validate WorldTrace's potential as a foundation dataset and UniTraj's consistent superiority in trajectory recovery tasks, with substantial improvements through fine-tuning, reinforcing its adaptability and robustness.

Trajectory Prediction. Table 3 shows UniTraj's exceptional performance in trajectory prediction, a different task requiring forward inference rather than reconstruction. The zero-shot results merit particular attention, as they represent the most challenging scenario for trajectory models. On WorldTrace, UniTraj's zero-shot MAE significantly outperforms all baselines, underscoring the model's versatility in capturing universal motion patterns. When fine-tuned, the performance further improves, consistently achieving the best results across all


Table 3: Performance comparison of UniTraj with trajectory prediction tasks.


<table><tr><td rowspan="2">Methods</td><td colspan="2">WorldTrace</td><td colspan="2">Chengdu</td><td colspan="2">GeoLife</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Linear</td><td>153.12</td><td>159.65</td><td>156.85</td><td>164.58</td><td>189.02</td><td>201.34</td></tr><tr><td>DHTR</td><td>146.48</td><td>151.63</td><td>123.47</td><td>129.73</td><td>180.32</td><td>187.59</td></tr><tr><td>Transformer</td><td>114.25</td><td>117.07</td><td>67.38</td><td>70.86</td><td>165.02</td><td>170.84</td></tr><tr><td>DeepMove</td><td>55.69</td><td>58.67</td><td>36.31</td><td>39.10</td><td>116.46</td><td>123.20</td></tr><tr><td>TrajBERT</td><td>80.57</td><td>86.36</td><td>64.73</td><td>68.92</td><td>113.68</td><td>121.18</td></tr><tr><td>TrajFM</td><td>75.45</td><td>81.32</td><td>77.82</td><td>80.48</td><td>121.94</td><td>128.16</td></tr><tr><td>UniTraj (zero-shot)</td><td>49.85</td><td>55.02</td><td>42.75</td><td>45.93</td><td>108.35</td><td>133.60</td></tr><tr><td>Improvement(%)</td><td>↑10.49</td><td>↑6.22</td><td>↓17.74</td><td>↓17.46</td><td>↑4.69</td><td>↓10.25</td></tr><tr><td>UniTraj (fine-tune)</td><td>30.10</td><td>34.46</td><td>28.78</td><td>32.44</td><td>90.97</td><td>102.88</td></tr><tr><td>Improvement(%)</td><td>↑45.95</td><td>↑41.27</td><td>↑20.74</td><td>↑17.03</td><td>↑19.98</td><td>↑15.10</td></tr></table>

evaluated datasets. This generalization capability stems from our Last-N masking strategy, which explicitly shapes the embedding space to support predictive inference. These results further confirm that UniTraj not only generalizes remarkably well across diverse datasets but also benefits considerably from fine-tuning, making it highly adaptable for real-world applications requiring accurate trajectory predictions.

Trajectory Classification. Figure 2 presents classification accuracy results that reveal UniTraj's capacity to learn discriminative representations of movement modalities. Notably, even without fine-tuning, UniTraj achieves  $71.3\%$  accuracy on GeoLife, outperforming several supervised baselines. This zero-shot performance demonstrates that the pre-trained representations inherently capture transportation mode signatures, where movement modality emerges as a natural organizing principle. On the Grab-Posisi

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/8c00cc2bc0e7ff21015487c83a5cf6be47209fab15c62b0870c27478bcef4c21.jpg)



Figure 2: Performance comparison of classification task with GeoLife and Grab-posisi dataset.


dataset, which presents additional challenges due to similar motion patterns for mixed travel modes (car and motorcycle). UniTraj achieves  $79.3\%$  accuracy after fine-tuning with a substantial improvement over the best baseline. This improvement emphasizes UniTraj's ability to capture subtle kinematic signatures that differentiate travel modes with complex or similar patterns.

# 5.3 Dataset Study

This section analyzes the impact of dataset scale, quality, and diversity on model performance of UniTraj, particularly its generalization capability across different data sources. We focus on two main experiments: (1) examining the effect of dataset scale and quality within WorldTrace, with varying data volumes ( $\{0.01, 0.5, 1\}$  millions) and a high-quality (obtained by further removing loops, staying dense trajectories) subset, and (2) assessing UniTraj's adaptability and effectiveness by training it on these datasets beyond WorldTrace, thus showing its potential as a foundation model.

Effect of Dataset Scale and Quality. Figure 3(a) illustrates the relationship between training data volume and model performance, revealing a phenomenon that goes beyond simple scaling laws. With increasing trajectory count from WorldTrace (from 0.5M to 2.45M), the MAE on the in-domain test set decreases dramatically, showing substantial improvement up to approximately 1M trajectories before beginning to exhibit diminishing returns. The above result indicates that larger datasets enable


Table 4: Ablation study on different resampling and masking strategies on six datasets.


<table><tr><td rowspan="2">Methods</td><td colspan="2">WorldTrace</td><td colspan="2">Chengdu</td><td colspan="2">Xi&#x27;an</td><td colspan="2">GeoLife</td><td colspan="2">Grab-posisi</td><td colspan="2">Porto</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>w/o Dynamic Multi-scale resampl.</td><td>426.80</td><td>482.37</td><td>192.54</td><td>272.42</td><td>157.85</td><td>223.96</td><td>499.95</td><td>671.69</td><td>1933.28</td><td>2504.16</td><td>93.14</td><td>119.93</td></tr><tr><td>w/o Interval Consistent resampl.</td><td>21.30</td><td>24.76</td><td>12.98</td><td>20.61</td><td>9.34</td><td>13.90</td><td>69.41</td><td>115.33</td><td>102.45</td><td>149.60</td><td>1724.12</td><td>2016.61</td></tr><tr><td>w/o Key points masking</td><td>25.49</td><td>28.91</td><td>14.46</td><td>21.98</td><td>11.10</td><td>15.17</td><td>45.94</td><td>72.84</td><td>113.65</td><td>162.57</td><td>76.51</td><td>101.18</td></tr><tr><td>w/o Block masking</td><td>7.79</td><td>10.47</td><td>9.22</td><td>15.36</td><td>7.16</td><td>11.18</td><td>48.59</td><td>77.73</td><td>89.34</td><td>128.72</td><td>198.41</td><td>238.88</td></tr><tr><td>UniTraj</td><td>10.22</td><td>13.56</td><td>11.98</td><td>20.94</td><td>8.93</td><td>13.83</td><td>37.21</td><td>63.89</td><td>114.07</td><td>167.01</td><td>78.28</td><td>100.14</td></tr></table>

the model to capture a wider range of spatio-temporal patterns. However, while increasing the dataset size from 1 million to 2.45 million trajectories results in better coverage, the model's MAE slightly increases due to the introduction of more noise in the full dataset. In contrast, training on a high-quality subset of 1 million trajectories, which includes curated, noise-free data, yields more reliable and consistent learning. This highlights the importance of both dataset scale and quality, with quality being especially crucial when data volume is limited.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/c005681dcd71072516bce4a17708a1592092f1c81bb0b2956018223d218c4bbe.jpg)



(a) Training data volume.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/21bd5e959f94d0d3ef471bdfc4407be33c724e20c2a5fdd65c6a76c159357664.jpg)



(b) Training dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/aeab726ae5756239152e6c4da6710a7cc1696b02ad33c32866e355891bc1544a.jpg)



(c) Number of Encoders


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/2c6eeefe375c20b5707ccbb667a5c5ff2a4eb8df549792b3bdc7e170d76c831f.jpg)



(d) Mask ratio  $(\%)$ .



Figure 3: The effect of amount of data volume, diversity dataset, and different parameter settings.


Effect of Dataset Diversity. In Figure 3(b), we compare UniTraj's zero-shot performance when trained on WorldTrace and Chengdu (the highest quality dataset available), evaluated across multiple real-world datasets. Models trained on WorldTrace exhibit superior generalization across diverse datasets (e.g., GeoLife and Porto), reflecting the broad geographic and contextual coverage of WorldTrace. Conversely, models trained on Chengdu perform best on datasets with similar density and travel modes, such as Xi'an. However, proprietary datasets like Chengdu, while offering high quality, are not publicly available, limiting their applicability for universal tasks. These results demonstrate UniTraj's robustness and adaptability, validating WorldTrace as an ideal training resource for building a universal trajectory foundation model. At the same time, the findings confirm that UniTraj can effectively leverage other datasets when necessary, further enhancing its versatility.

# 5.4 Model Study

We investigate architectural components, parameter settings, and pre-training strategies to assess sensitivity to parameter choices and the contributions of their core components.

Effect of Parameter Settings. Figure 3(c) Figure 3(d) and presents the results of our parameter sensitivity analysis, examining how the number of encoder blocks and the masking ratio influence model performance. As shown in Figure 3(c), increasing the number of encoder blocks from 2 to 8 significantly reduces MAE, with performance plateauing beyond 8 blocks. This plateau suggests that while deeper architectures can improve model capacity, the benefits diminish without corresponding adjustments in data or hyperparameters [17]. Figure 3(d) demonstrates that a masking ratio of  $50\%$  yields the best performance. Low masking ratios (e.g.,  $5\% -10\%$ ) result in higher MAE due to insufficient training signal, while higher ratios (e.g.,  $75\%$ ) lead to increased MAE from excessive information loss. A  $50\%$  masking ratio strikes a balance, providing the model with a strong training signal without sacrificing the context needed for effective trajectory reconstruction.

Ablation Study. Table 4 presents an ablation study, showing how different pre-training strategies affect UniTraj's performance across datasets. The performance varies across datasets, indicating the effectiveness and limitations of them depending on the specific data and task scenarios. Dynamic Multi-scale Resampling significantly improves performance across most datasets, especially GeoLife and Grab-Posisi, which have inconsistent sampling intervals and lower data quality. This suggests that dynamic resampling helps the model to adapt to heterogeneous dataset scenarios and to be adaptive for information preservation (see Appendix B.1.1 for more details). The Interval Consistent Resampling has a notable positive effect on datasets with consistent sampling rates, such as Porto

and WorldTrace. It indicates that the integration of this strategy strategy can effectively separate the temporal sampling pattern from the region, it enhances the generalization of the model to data sets with different sampling rates (analysis presented in Appendix B.1.2). Key Points Masking leads to substantial performance drops on high-quality datasets like Chengdu and Xi'an but appears to offer minimal benefits, or even slight disadvantages, for certain datasets. This finding suggests that adjusting adaptive masking strategies based on trajectory complexity, potentially applying it selectively to trajectories with significant directional changes, while using alternative strategies for smoother paths. Block Masking shows significant effects on GeoLife and Porto, where it helps the model handle low sampling frequencies. However, its impact on other datasets is more inconsistent, suggesting that it introduces an artificial challenge that may increase complexity in high-frequency datasets. (we provide a robustness analysis in Appendix B.2) Overall, the varying impact of UniTraj's pre-training strategies across datasets highlights its adaptability to different tasks and scenarios. While not all of them universally enhance performance, their combined use provides a balanced training strategy, allowing for flexible configuration depending on specific dataset requirements. Fine-tuning further optimizes performance, ensuring stability and robustness across diverse tasks.

# 6 Conclusion

In this work, we presented UniTraj, a universal trajectory foundation model designed to overcome the task specificity, regional dependency, and data quality limitations of current approaches. UniTraj acts as a robust backbone that generalizes effectively across diverse tasks and regions. To support its development, we introduced WorldTrace, a high-quality global dataset with 2.45 million trajectories from 70 countries, offering broad geographic coverage, varied sampling rates, and open accessibility. Together, UniTraj and WorldTrace provide a versatile, high-performing foundation for trajectory analysis, paving the new solution for more adaptable and efficient models in trajectory-based research. Future work will focus on expanding the geographic and modal diversity of the WorldTrace dataset to better cover underrepresented regions and non-motorized travel. We also aim to enhance the UniTraj model by integrating contextual information, such as road networks and points of interest, to improve its predictive accuracy and real-world applicability. Further optimizations to the model architecture and pre-training strategies will also be explored to boost performance and efficiency.

# References



[1] T. B. Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.





[2] Y. Chang, E. Tanin, G. Cong, C. S. Jensen, and J. Qi. Trajectory similarity measurement: An efficiency perspective. arXiv preprint arXiv:2311.00960, 2023.





[3] W. Chen, Y. Liang, Y. Zhu, Y. Chang, K. Luo, H. Wen, L. Li, Y. Yu, Q. Wen, C. Chen, et al. Deep learning for trajectory data management and mining: A survey and beyond. arXiv preprint arXiv:2403.14151, 2024.





[4] C. Chu, H. Zhang, and F. Lu. Trajgdm: A new trajectory foundation model for simulating human mobility. In Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems, pages 1-2, 2023.





[5] S. Dabiri and K. Heaslip. Inferring transportation modes from GPS trajectories using a convolutional neural network. Transportation research part C: emerging technologies, 86:360-371, 2018.





[6] A. Das, W. Kong, R. Sen, and Y. Zhou. A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023.





[7] J. Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.





[8] Didi Chuxing. Gaia open datasets., 2018.





[9] A. Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.





[10] D. H. Douglas and T. K. Peucker. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica: the international journal for geographic information and geovisualization, 10(2):112-122, 1973.





[11] J. Feng, Y. Li, C. Zhang, F. Sun, F. Meng, A. Guo, and D. Jin. Deepmove: Predicting human mobility with attentional recurrent networks. In Proceedings of the 2018 world wide web conference, pages 1459-1468, 2018.





[12] C. Guo, B. Yang, J. Hu, and C. Jensen. Learning to route with sparse trajectory sets. In 2018 IEEE 34th International Conference on Data Engineering (ICDE), pages 1073-1084. IEEE, 2018.





[13] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.





[14] B. Hofmann-Wellenhof, H. Lichtenegger, and E. Wasle. GNSS-global navigation satellite systems: GPS, GLONASS, Galileo, and more. Springer Science & Business Media, 2007.





[15] X. Huang, Y. Yin, S. Lim, G. Wang, B. Hu, J. Varadarajan, S. Zheng, A. Bulusu, and R. Zimmermann. Grab-posisi: An extensive real-life gps trajectory dataset in southeast asia. In Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Prediction of Human Mobility, page 1-10, 2019.





[16] J. Jiang, D. Pan, H. Ren, X. Jiang, C. Li, and J. Wang. Self-supervised trajectory representation learning with temporal regularities and travel semantics. In 2023 IEEE 39th international conference on data engineering (ICDE), pages 843-855. IEEE, 2023.





[17] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.





[18] H. Lan, J. Xie, Z. Bao, F. Li, W. Tian, F. Wang, S. Wang, and A. Zhang. Vre: a versatile, robust, and economical trajectory data system. Proceedings of the VLDB Endowment, 15(12):3398-3410, 2022.





[19] L. Li, R. Jiang, Z. He, X. M. Chen, and X. Zhou. Trajectory data-based traffic flow studies: A revisit. Transportation Research Part C: Emerging Technologies, 114:225-240, 2020.





[20] Z. Li, L. Xia, L. Shi, Y. Xu, D. Yin, and C. Huang. Opencity: Open spatio-temporal foundation models for traffic prediction. arXiv preprint arXiv:2408.10269, 2024.





[21] Y. Liang, K. Ouyang, Y. Wang, X. Liu, H. Chen, J. Zhang, Y. Zheng, and R. Zimmermann. Trajformer: Efficient trajectory classification with transformers. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 1229-1237, 2022.





[22] Y. Lin, H. Wan, S. Guo, J. Hu, C. S. Jensen, and Y. Lin. Pre-training general trajectory embeddings with maximum multi-view entropy coding. IEEE Transactions on Knowledge and Data Engineering, 36(12):9037-9050, 2023.





[23] Y. Lin, T. Wei, Z. Zhou, H. Wen, J. Hu, S. Guo, Y. Lin, and H. Wan. Trajfm: A vehicle trajectory foundation model for region and task transferability. arXiv:2408.15251, 2024.





[24] Y. Lin, Z. Zhou, Y. Liu, H. Lv, H. Wen, T. Li, Y. Li, C. S. Jensen, S. Guo, Y. Lin, et al. Unite: A survey and unified pipeline for pre-training st trajectory embeddings. arXiv e-prints, pages arXiv-2407, 2024.





[25] M. Luca, G. Barlacchi, B. Lepri, and L. Pappalardo. A survey on deep learning for human mobility. ACM Computing Surveys (CSUR), 55(1):1-44, 2021.





[26] Z. Ma, Z. Tu, X. Chen, Y. Zhang, D. Xia, G. Zhou, Y. Chen, Y. Zheng, and J. Gong. More than routing: Joint GPS and route modeling for refine trajectory representation learning. In Proceedings of the ACM Web Conference 2024, pages 3064–3075, 2024.





[27] G. Mai, W. Huang, J. Sun, S. Song, D. Mishra, N. Liu, S. Gao, T. Liu, G. Cong, Y. Hu, et al. On the opportunities and challenges of foundation models for geospatial artificial intelligence. arXiv preprint arXiv:2304.06798, 2023.





[28] W. K. Meghan O'Connell, moreiraMatias. Ecml/pkdd 15: Taxi trajectory prediction (i), 2015.





[29] T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, and A. Grover. Climax: A foundation model for weather and climate. arXiv preprint arXiv:2301.10343, 2023.





[30] OpenStreetMap Contributors. Openstreetmap, 2024.





[31] J. Si, J. Yang, Y. Xiang, H. Wang, L. Li, R. Zhang, B. Tu, and X. Chen. Trajbert: Bert-based trajectory recovery with spatial-temporal refinement for implicit sparse trajectories. IEEE Transactions on Mobile Computing, 2023.





[32] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.





[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.





[34] G. Wang, X. Chen, F. Zhang, Y. Wang, and D. Zhang. Experience: Understanding long-term evolving patterns of shared electric vehicle networks. In The 25th Annual international conference on mobile computing and networking, pages 1-12, 2019.





[35] J. Wang, N. Wu, X. Lu, W. X. Zhao, and K. Feng. Deep trajectory recovery with fine-grained calibration using kalman filter. IEEE Transactions on Knowledge and Data Engineering, 33(3):921-934, 2019.





[36] S. Wang, Z. Bao, J. S. Culpepper, and G. Cong. A survey on trajectory data management, analytics, and learning. ACM Computing Surveys (CSUR), 54(2):1-36, 2021.





[37] Z. Wang, K. Fu, and J. Ye. Learning to estimate the travel time. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, page 858-866, London, United Kingdom, 2018. Association for Computing Machinery.





[38] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo. Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592, 2024.





[39] H. Yan and Y. Li. Generative ai for intelligent transportation systems: Road transportation perspective. ACM Computing Surveys, 2025.





[40] C. Yang and G. Gidofalvi. Fast map matching, an algorithm integrating hidden markov model with precomputation. International Journal of Geographical Information Science, 32(3):547-570, 2018.





[41] X. Yu, J. Wang, Y. Yang, Q. Huang, and K. Qu. Bigcity: A universal spatiotemporal model for unified trajectory and traffic state data analysis. arXiv preprint arXiv:2412.00953, 2024.





[42] J. Yuan, Y. Zheng, C. Zhang, W. Xie, X. Xie, G. Sun, and Y. Huang. T-drive: Driving directions based on taxi trajectories. In Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems, GIS '10, page 99-108, New York, NY, USA, 2010. Association for Computing Machinery.





[43] Y. Yuan, J. Ding, J. Feng, D. Jin, and Y. Li. Unist: a prompt-empowered universal model for urban spatio-temporal prediction. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4095–4106, 2024.





[44] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhoff. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pages 2114–2124, 2021.





[45] P. Zhao, A. Luo, Y. Liu, J. Xu, Z. Li, F. Zhuang, V. S. Sheng, and X. Zhou. Where to go next: A spatio-temporal gated network for next poi recommendation. IEEE Transactions on Knowledge and Data Engineering, 34(5):2512-2524, 2020.





[46] Y. Zheng, H. Fu, X. Xie, W.-Y. Ma, and Q. Li. Geolife GPS trajectory dataset - User Guide, July 2011.





[47] Y. Zheng, L. Zhang, X. Xie, and W.-Y. Ma. Mining interesting locations and travel sequences from gps trajectories. In Proceedings of the 18th international conference on World wide web, pages 791-800, 2009.





[48] S. Zhou, S. Shang, L. Chen, C. S. Jensen, and P. Kalnis. Red: Effective trajectory representation learning with comprehensive information. arXiv preprint arXiv:2411.15096, 2024.





[49] Y. Zhu, Y. Ye, Y. Wu, X. Zhao, and J. J. Yu. Synmob: Creating high-fidelity synthetic GPS trajectory dataset for urban mobility analysis. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.





[50] Y. Zhu, J. J. Yu, X. Zhao, Q. Liu, Y. Ye, W. Chen, Z. Zhang, X. Wei, and Y. Liang. Controltraj: Controllable trajectory generation with topology-constrained diffusion model. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4676-4687, 2024.



# SUPPLEMENTARY MATERIAL

UNITRAJ: LEARNING A UNIVERSAL TRAJECTORY FOUNDATION MODEL FROM BILLION-SCALE WORLDWIDE TRACES

TABLE OF CONTENTS

# A Details of WorldTrace Dataset 15

A.1 Data Collection 15

A.2 Data Processing 16

A.3 Data Statistics and Analysis 16

A.4 Data Privacy and Copyright 17

# B Pre-training Strategies 18

B.1 Adaptive Trajectory Resampling 18

B.2 Self-supervised Trajectory Masking 21

# C Details of UniTraj 24

C.1 Overall Architecture 24

C.2 Input Representation and Embedding 24

C.3 Adaptive Representation Learning 25

C.4 Task-Specific Adaptation 26

C.5 Implementation Details 27

# D Experiments Details 27

D.1 Datasets 27

D.2 Tasks Applicability Study Settings 28

D.3 Dataset Study Settings 30

D.4 Model Study Settings 30

# E More Discussion 30

E.1 Limitation 30

E.2 Broader Impact 31

# A Details of WorldTrace Dataset

In this section, we detail the collection of the dataset, the processing, and provide a detailed analysis of the resulting dataset.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/59e20a8787a80f70cf017ed6bee0d2982b0ae21281163284a9f1cbc929302dfe.jpg)



Figure 4: The process pipeline of WorldTrace dataset construction.


# A.1 Data Collection

Data Source. As shown in Figure 4, the raw data for WorldTrace is sourced from the shared trajectory data platform on OpenStreetMap (OSM) [30]5. This platform, a public sharing project, hosts over 11 million GPS trajectories uploaded by contributors worldwide from 2004 to the present. To ensure data quality and reliability, we specifically targeted contributions tagged for motorized movement to ensure data currency and relevance to modern transportation networks. This approach helps minimize device heterogeneity and avoids outdated data that might not reflect current infrastructure. The raw data is stored in the standardized GPX (GPS Exchange Format), an XML schema designed for exchanging GPS data between applications and web services https://www.topografia.com/GPX/1/1/. Each GPX file contains sequences of trackpoints with the following attributes:

- Latitude (decimal degrees)

- Longitude (decimal degrees)

- Altitude (decimal numbers)

- Timestamp (ISO 8601 format)

- Optional metadata (version, tags, etc.)

In addition, while crawling the original trajectory, we also crawled the basic information about the trajectory descriptions, such as the starting point, markers, time, creator, etc., which was saved as a JSON file.

Collection Process. Prior to integration, our collection pipeline involved the following steps: Our collection pipeline involved the following steps:

1. API-based Retrieval: We use the OSM API to systematically query and download GPX traces based on selected filters to ensure global coverage. In order not to increase the burden on server providers, we did not use concurrent crawling, and the whole collection process lasted about 6 months, yielding about 4.5 million raw traces.

2. Initial Filtering: During acquisition, we implemented preliminary filtering to exclude trajectories with obvious anomalies such as: Coordinates outside valid ranges  $(-90^{\circ}$  to  $90^{\circ}$  for latitude,  $-180^{\circ}$  to  $180^{\circ}$  for longitude); Duplicate or long duration consecutive points; Empty or near-empty traces (fewer than 60 seconds).

3. Format Standardization: All collected data was parsed from the original GPX format and converted to a unified internal format for subsequent processing.

# A.2 Data Processing

Our preprocessing pipeline was designed to balance preserving authentic movement patterns with removing noise and inconsistencies. The process consists of three main stages:

Normalization. The raw data exhibited highly variable sampling frequencies, ranging from subsecond intervals up to several seconds between consecutive points. This heterogeneity creates challenges for modeling and increases storage requirements unnecessarily. We therefore applied the following normalization procedures:

- Temporal Resampling: We resampled all trajectories to a uniform rate of one point per second (1 Hz). For segments with sampling rates higher than 1 Hz, we select the first occurrence of a trajectory point within each one-second window. For segments with lower sampling rates, we used linear interpolation between available points to estimate positions at one-second intervals.

- Coordinate Standardization: All coordinates were converted to the WGS84 datum for consistency, and we ensured uniform precision across the dataset (6 decimal places for both latitude and longitude, providing 0.1m precision at the equator).

Filtering. After normalization, we implemented a multi-stage filtering process to meticulously remove trajectories that were deemed unsuitable for our analysis. This comprehensive filtering approach involved several key steps:

- Length-based Filtering: We discarded trajectories with fewer than 32 points (equivalent to 32 seconds after resampling) or covering distances below 100 meters, as these typically represent stationary periods or very short movements with limited analytical value.

- Speed-based Filtering: We calculated point-to-point speeds and removed trajectories containing implausible values (e.g., exceeding  $120\mathrm{km/h}$  or lower  $0.5\mathrm{km/h}$  in urban environments), typically caused by GPS errors or anomalies.

- Distance-based Outlier Detection: We calculated the distance between the original trajectory and the map-matched trajectory. Trajectories that were too far away (indicating large deviations in motion) were flagged for further inspection or removal.

- Loop Detection: We identify and remove trajectories that form perfect or near-perfect loops with no apparent destination by their geometry, which usually indicates the presence of clearly anomalous patterns.

Calibration. GPS signals can suffer from various errors due to atmospheric conditions, satellite geometry, and physical obstructions. To improve data quality, we applied map-matching techniques to align raw GPS points with underlying road networks, using a Hidden Markov Model-based approach (or using online API) with a custom emission probability function that accounts for both point-to-road distance and heading consistency. Besides, each trajectory point was enriched with derived attributes.

# A.3 Data Statistics and Analysis

Overall Statistics. The final WorldTrace dataset contains:

- Approximately 2.45 million trajectories.

8.8 billion raw GPS points (before normalization).

- Coverage across 70 countries on all inhabited continents.

- Temporal span from August 2021 to December 2023.

- Average trajectory duration of approximately 6 minutes.

- Average trajectory distance of 5.73 kilometers.

Average travel speed of  $48.0\mathrm{km / h}$

- Points per trajectory ranging from 32 to over 600, with an average of 358 points.

Geographic Distribution. WorldTrace offers extensive geographic coverage, as illustrated in Figure 5, encompassing trajectory data from 70 countries and spanning diverse environments and

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/47ae54c1d6ad01046bb8440aaab58a4d40b91936b42a8ab6d35f7f3e3bb6f84b.jpg)



(a) Geographic distribution across the world.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/30450094f5cf9c0d7f3fb82895f6c5543d01d4fb272d9d1147bf32e445e5cc91.jpg)



(b) Trajectory counts of top 10 countries.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/6654e7d1d483534d72bb57d45ddcb600fb2e11ffd8b4fdc7591fb21735783bca.jpg)



(c) Distribution within contiguous USA.



Figure 5: The distribution details of WorldTrace dataset.


infrastructure types. This global distribution is visualized in Figure 5(a), highlighting dense concentrations in North America, East Asia, and parts of Europe, with trajectory counts exceeding in the most represented regions. Figure 5(b) further details the top 10 countries by trajectory counts, with the United States, China, and Canada leading in data volume. Notably, it exhibits substantial geographic diversity, with varying densities across urban, suburban, and rural environments. The top 10 countries by trajectory count, namely, the USA, China, Canada, Germany, UK, Japan, Brazil, Australia, South Korea, and Hungary, represent a wide range of urban forms, road networks, and mobility cultures. Additionally, Figure 5(c) provides a closer look at the data density within the contiguous United States, demonstrating high-resolution coverage along major road networks and urban centers. This detailed distribution underscores the dataset's ability to capture nuanced variations in trajectory data across different regions. Collectively, these figures emphasize the potential of WorldTrace to serve as a robust foundation for developing region-independent and universal trajectory models. Its extensive geographic coverage and diverse environmental representation make it well-suited for applications that require broad and adaptable trajectory data.

# A.4 Data Privacy and Copyright

To protect privacy and comply with international data protection regulations, all data collection adhered strictly to privacy regulations and ethical guidelines. Trajectories were anonymized, and any personally identifiable information was excluded to protect user privacy. In addition, all raw data follows the Open Data Commons Open Database License (ODbL) license from OSM: http://opendatacommons.org/licenses/odbl/1.0/. We will share derived datasets under the same license terms to respect the data use policies of the community.

# B Pre-training Strategies

In this section, we provide specific details on the adaptive trajectory resampling strategy and the self-supervised trajectory masking strategy, and we will provide the design motivation and theoretical analysis for these two strategies.

# B.1 Adaptive Trajectory Resampling

Trajectory data heterogeneous is one of the main challenges in cross-regional and cross-device trajectory modeling. The Adaptive Trajectory Resampling strategy solves this problem through two complementary components: Dynamic Multi-Scale Resampling and Interval Consistent Resampling. We designed these two strategies with the motivation of fitting different regions and dataset qualities through diversified trajectory sampling frequencies and motion patterns. Dynamic Multi-Scale Resampling ensures an optimal balance between information preservation and computational efficiency across different trajectory lengths, prioritizing the retention of key motion patterns. Interval Consistent Resampling enhances the model's generalization ability across datasets with different sampling rates by normalizing the time dimension.

# B.1.1 Dynamic Multi-Scale Resampling

As discussed in Section 4.2, we adopted a logarithmic resampling ratio that adjusts the sampling rate according to the trajectory length. The resampling ratio function  $R(n)$  is designed to decrease logarithmically as the trajectory length  $n$  increases:

$$
R (n) = \left\{ \begin{array}{l l} R _ {\min }, & n \geq n _ {\max } \\ 1 - \left(1 - R _ {\min }\right) \phi (n), & n _ {\min } <   n <   n _ {\max } \\ 1, & n \leq n _ {\min } \end{array} \right. \tag {7}
$$

where  $R_{\mathrm{min}}$  is the minimum sampling ratio, and  $n_{\mathrm{min}}$  and  $n_{\mathrm{max}}$  denotes the shortest and longest length thresholds, respectively. The normalization factor  $\phi(n)$  is computed as follows:

$$
\phi (n) = \frac {\ln \left(n - n _ {\min } + 1\right)}{\ln \left(n _ {\max } - n _ {\min } + 1\right)}. \tag {8}
$$

Formal Definition. Here, we provide a formal definition and theoretical analysis of the above empirical results through information theory and computational efficiency perspectives. For any trajectory  $\pmb{\tau} = \{p_1, p_2, \dots, p_n\}$  consists of  $n$  spatio-temporal points. The number of points for resampled trajectory  $\pmb{\tau}' = \{p_1, p_2, \dots, p_m\}$  is:

$$
m = R (n) \cdot n, \tag {9}
$$

where function  $R(n)$  that determines what proportion of points to retain. The logarithmic sampling strategy guarantees bounded sample sizes for arbitrarily long trajectories while preserving critical minimum information content. Specifically:

- For  $n \leq n_{\min}$ :  $R(n) = 1$ , so  $m = n$ ;

- For  $n \geq n_{\max}$ :  $R(n) = R_{\min}$ , we set  $m = m_{\max}$  as a constant. Clearly, the number of sampled points is bounded above by  $m_{\max}$ .

To ensure boundedness, we analyze  $m(n)$  in the intermediate domain  $n \in (n_{\min}, n_{\max})$ .

$$
m = \left[ 1 - \left(1 - R _ {\min }\right) \cdot \phi (n) \right] \cdot n. \tag {10}
$$

Taking derivative:

$$
\frac {d (R (n) \cdot n)}{d n} = 0 \tag {11}
$$

Solving this equation yields a value  $n^* < n_{\max}$ , ensuring that  $m_{\max}$  is bounded. Since  $R(n)$  becomes constant for  $n \geq n_{\max}$ , and  $m$  increases linearly in that region, the global maximum occurs at either  $n^*$  or  $n_{\max}$ . However, due to the logarithmic decay of  $R(n)$ , the growth of  $m$  slows, and the maximum value is achieved at a finite  $n^* < n_{\max}$ . Hence,  $m$  is bounded for all  $n$ .

# Corollary 1: Information Preservation and Computing Efficiency Optimization

**Standpoint:** The logarithmic sampling function provides an optimal balance between information preservation and computational efficiency across varying trajectory lengths.

Proof: Let  $I(\tau)$  represent the information content of trajectory  $\tau$ . Empirical studies in spatiotemporal data analysis suggest that information content typically scales sub-linearly with trajectory length, following approximately:

$$
I (\boldsymbol {\tau}) \propto n ^ {\alpha}, \tag {12}
$$

where  $0 < \alpha < 1$ . For example,  $\alpha \approx 0.7$  indicates that only  $70\%$  of the trajectory points contain valid feature information, and the remaining  $30\%$  are redundant. For a resampled trajectory  $\tau'$  with  $m = R(n) \cdot n$  points, the information preservation ratio  $\eta$  can be approximated as:

$$
\eta = \frac {I \left(\boldsymbol {\tau} ^ {\prime}\right)}{I (\boldsymbol {\tau})} \approx \left(\frac {m}{n}\right) ^ {\alpha} = R (n) ^ {\alpha}. \tag {13}
$$

The computational cost  $C$  of processing trajectory typically scales linearly with length:

$$
C (\tau) \propto n ^ {\beta}, \tag {14}
$$

where  $\beta \geq 1$ , typically  $\beta \approx 2$  for Transformer-based models. After resampling, the computational efficiency gain  $\gamma$  is:

$$
\gamma = \frac {C (\boldsymbol {\tau})}{C \left(\boldsymbol {\tau} ^ {\prime}\right)} \approx \left(\frac {n}{m}\right) ^ {\beta} = \frac {1}{R (n) ^ {\beta}}. \tag {15}
$$

The optimal sampling function maximizes the product of information preservation and computational efficiency:

$$
\max  _ {R (n)} \eta \cdot \gamma = \max  _ {R (n)} R (n) ^ {\alpha} \cdot \frac {1}{R (n) ^ {\beta}} = \max  _ {R (n)} R (n) ^ {\alpha - \beta}. \tag {16}
$$

Since  $\alpha < \beta$  for typical trajectory data, this is a decreasing function  $R(n)$ . However, we must maintain a minimum level of information, hence the constraint  $R(n) \geq R_{\min}$ .

When we examine the information density:

$$
D (n) = \frac {I (\tau)}{n} \propto n ^ {\alpha - 1}, \tag {17}
$$

we observe that it decreases as  $n$  increases, indicating diminishing information return per point in longer trajectories. An optimal sampling ratio should proportionally track this information density:

$$
R _ {\text {o p t}} (n) \propto D (n) \propto n ^ {\alpha - 1}. \tag {18}
$$

Our logarithmic resampling function's derivative in the intermediate domain  $(n_{\mathrm{min}} < n < n_{\mathrm{max}})$  is:

$$
\frac {d R (n)}{d n} = - \frac {1 - R _ {\operatorname* {m i n}}}{\ln \left(n _ {\operatorname* {m a x}} - n _ {\operatorname* {m i n}} + 1\right)} \cdot \frac {1}{n - n _ {\operatorname* {m i n}} + 1} \propto \frac {1}{n} \tag {19}
$$

As  $n$  increases, the growth rate of the logarithmic function slows down, causing the rate at which the sampling rate  $R(n)$  decreases to also slow down. This is closely proportional to the derivative of the theoretical optimal sampling rate:

$$
\frac {d R _ {\mathrm {o p t}} (n)}{d n} \propto (\alpha - 1) n ^ {\alpha - 2} \propto \frac {1}{n ^ {2 - \alpha}}. \tag {20}
$$

For example, when  $\alpha \approx 0.7$ , we have  $\frac{dR_{\mathrm{opt}}(n)}{dn} \propto \frac{1}{n^{1.3}}$ . This property of logarithmic functions (their rate of change is inversely proportional to the input value), making them naturally suited to this task. Therefore, Logarithmic resampling provides a theoretically reasonable compromise: it preserves almost all of the information from short trajectories (where every point may be significant) while reducing redundancy in long trajectories (where redundancy is highest). Compared to linear functions, logarithmic functions can more naturally adapt to the information density curve across the entire trajectory length range.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/1f87901cb9a3a0995221fb57285f48e0646ae4e96d0b5e4b8f54cf4373b6840e.jpg)



Figure 6: Illustration of the difference between dynamic resampling strategies with linear method.


Visualization. As shown in Figure 6, we compare in detail the proposed dynamic resampling strategy with a linear resampling strategy (where the sampling ratio  $R(n)$  decreases linearly with the length of the trajectory) regarding the sampling ratio and the sampled points. Specifically, this figure illustrates the dynamic resampling strategy compared to a linear resampling approach. The top plot displays how the sampling ratio  $R(n)$  decreases with trajectory length  $n$ . The dynamic strategy (orange curve) follows a logarithmic decrease, ensuring a smoother transition from retaining all points for short trajectories ( $n \leq n_{\mathrm{min}}$ ) to reducing redundancy for long trajectories ( $n \geq n_{\mathrm{max}}$ ), with a minimum sampling ratio  $R_{\mathrm{min}}$ . In contrast, the linear resampling strategy (blue curve) decreases the sampling ratio at a constant rate. The bottom plot shows the relationship between the number of sampled points and trajectory length for both strategies. The dynamic approach adjusts sampling more gradually, preserving detail for intermediate trajectories while minimizing redundancy in longer trajectories. However, linear sampling methods instead suffer from redundancy of sampling points due to the smoothly decreasing sampling rate. This dynamic resampling strategy ensures a balance between data volume reduction and the retention of critical movement details. The visual comparison highlights the adaptive nature of the dynamic strategy.

# B.1.2 Interval Consistent Resampling

Consider different cities may exhibit drastically different sampling intervals due to: Varying data collection protocols (e.g., 1s in City A vs. 5s in City B) and technical limitations or regional preferences in tracking technologies. This heterogeneity poses a serious challenge for developing universal trajectory models, as models trained on data from one region may fail to generalize to regions with different sampling characteristics. Therefore, we performed consistent interval sampling (at random time intervals) on the original dataset to ensure its generalizability across different datasets. Specifically, ICR standardizes the temporal intervals between trajectory points, transforming a trajectory  $\pmb{\tau} = \{(x_1,y_1,t_1),(x_2,y_2,t_2),\ldots ,(x_n,y_n,t_n)\}$  with irregular time intervals into a trajectory  $\pmb{\tau}' = \{(x_{1},y_{1},\Delta t),(x_{2},y_{2},\Delta t),\ldots ,(x_{m},y_{m},\Delta t)\}$  with uniform time intervals  $\Delta t = t_{i + 1} - t_i$ , for all  $i\in [1,m - 1]$ .

# Corollary 2: Temporal Regularity for Cross-Dataset Generalization

**Standpoint:** Interval consistent resampling regularizes the temporal dimension of trajectory samples, enhancing the model's ability to generalize across datasets with heterogeneous sampling rates.

Proof: Let  $\mathcal{D}_1$  and  $\mathcal{D}_2$  be two datasets of region with average sampling intervals  $\mu_{\Delta t}^{(1)}$  and  $\mu_{\Delta t}^{(2)}$ . Assume the temporal pattern recognition task can be formalized as learning a function  $f_{\theta}: \tau \to Y$  where the learned parameters  $\theta$  should ideally be robust to sampling rate variations. For trajectories with irregular sampling, the model must learn the relationship:

$$
y = f _ {\theta} \left(\left(x _ {1}, y _ {1}, t _ {1}\right), \left(x _ {2}, y _ {2}, t _ {2}\right), \dots , \left(x _ {n}, y _ {n}, t _ {n}\right)\right). \tag {21}
$$

This requires implicitly learning the distribution of time intervals  $P(\Delta T)$ , which varies across datasets. With ICR, the learning problem becomes:

$$
y = f _ {\theta} \left(\left(x _ {1} ^ {\prime}, y _ {1} ^ {\prime}, t _ {1} ^ {\prime}\right), \left(x _ {2} ^ {\prime}, y _ {2} ^ {\prime}, t _ {2} ^ {\prime}\right), \dots , \left(x _ {m}, y _ {m}, t _ {m}\right)\right), \quad \text {w i t h} t _ {i + 1} ^ {\prime} - t _ {i} ^ {\prime} = \Delta t _ {\text {f i x e d}} \tag {22}
$$

where temporal intervals are now consistently fixed, eliminating the need to learn dataset-specific temporal distributions.

Information Entropy Analysis. From the entropy perspective, consider trajectories from different regions  $r$  with characteristic sampling intervals  $\Delta t^r$ , where the distribution of intervals can be modeled as:

$$
P (\Delta t \mid r) \sim \mathcal {N} \left(\mu_ {r}, \sigma_ {r} ^ {2}\right), \tag {23}
$$

where  $\mathcal{N}$  is a dataset distribution with region-specific mean  $\mu_r$  and variance  $\sigma_r^2$ . The entropy of the joint distribution of regions (or dataset) and sampling intervals is:

$$
H (\mathcal {D}, \Delta T) = H (\mathcal {D}) + H (\Delta T \mid \mathcal {D}). \tag {24}
$$

This high conditional entropy  $H(\Delta T \mid \mathcal{D})$  creates a strong statistical correlation between regions and temporal patterns, forcing region-specific model adaptations. Interval Consistent Resampling transforms the original trajectory  $\pmb{\tau}$  into  $\pmb{\tau}'$  where  $t_{i+1}' - t_i' = \Delta t_{\mathrm{fixed}} \forall i \in [1,m-1]$ . This transformation minimizes the conditional entropy:

$$
H \left(\Delta T ^ {\prime} \mid \mathcal {D}\right) \approx 0, \tag {25}
$$

which effectively decoupling the temporal sampling pattern from the region. This transformation reduces dataset-specific temporal variability, thereby bringing the conditional distributions of trajectories across datasets closer in distributional space:

$$
P \left(\boldsymbol {\tau} ^ {\prime} \mid \mathcal {D} _ {1}\right) \approx P \left(\boldsymbol {\tau} ^ {\prime} \mid \mathcal {D} _ {2}\right). \tag {26}
$$

The reduction means the model sees more consistent input distributions, thus reducing the domain gap in learning.

For trajectory modeling tasks that focus on spatial patterns rather than absolute temporal dynamics, information loss is minimal when resampling preserves relative temporal order and approximate speed relationships. For a trajectory with velocity profile  $v(t) = (p_{i + 1} - p_i) / (t_{i + 1} - t_i)$ , the constraint:

$$
\frac {\left\| p _ {i + 1} ^ {\prime} - p _ {i} ^ {\prime} \right\|}{\left\| p _ {i + 1} - p _ {i} \right\|} \approx \frac {\Delta t _ {\text {f i x e d}}}{\Delta t _ {i}} \tag {27}
$$

ensures that relative speed information is preserved even as absolute time intervals are normalized.

# B.2 Self-supervised Trajectory Masking

Self-supervised Trajectory Masking (STM) forms a critical component of UniTraj's pre-training strategy, enabling the model to learn robust representations from incomplete trajectory data. While we introduced the concept in the main paper, this appendix provides a more detailed examination of the theoretical foundations, implementation details, and empirical justifications for our masking approach. Our Self-supervised Trajectory Masking framework implements four complementary masking strategies (as illustrated in Figure 7), each designed to simulate different types of real-world data incompleteness and encourage specific learning objectives:

# B.2.1 Random Masking

Random Masking applies a uniform probability distribution to select trajectory points for masking, where each point has an equal chance of being masked regardless of its position or significance. Formally, we select a subset of indices:

$$
\mathbf {I} _ {\text {r a n d}} \sim \operatorname {U n i f o r m} (\{1, 2, \dots , n \}). \tag {28}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/574b572f76feecda12f00dfa2501cb580c27e97c91fd45f4f27e0bef69f4f4d9.jpg)



Figure 7: Illustration of the difference masking strategies.


to mask. This strategy forces the model to develop both local and global dependencies, as it must learn to infer missing points from surrounding context without relying on predictable patterns. Random masking is a general masking strategy used to simulate sensor failures or temporary GPS signal loss that often occur in random trajectories.

# B.2.2 Block Masking

Block Masking conceals consecutive segments of the trajectory by selecting a starting point  $k$  and masking  $b$  consecutive points:

$$
\mathbf {I} _ {\text {b l o c k}} = \{k, k + 1, \dots , k + b - 1 \}, \text {f o r s o m e} k. \tag {29}
$$

This approach simulates extended sensor failures, tunnels, or urban canyons where trajectory data may be unavailable for continuous periods. The strategy challenges the model to reconstruct substantial missing segments by understanding the broader movement context, encouraging the development of long-range dependencies and trajectory continuity reasoning.

# B.2.3 Key Points Masking


Algorithm 1 Ramer-Douglas-Peucker (RDP) Algorithm


1: RDP(τ, s, e, ε)  
2: Initialize max distance  $d_{\mathrm{max}} \gets 0$   
3: Initialize index  $k \gets -1$   
4: for  $i = s + 1$  to  $e - 1$  do  
5: Calculate the distance from  $p_i$  to  $\overline{pspe}$ :  $d_i$   
6: if  $d_i > d_{\mathrm{max}}$  then  
7: Update max distance  $d_{\mathrm{max}} \gets d_i$   
8: Update index  $k \gets i$   
9: end if  
10: end for  
11: if  $d_{\mathrm{max}} > \epsilon$  then  
12:  $\tau_{\mathrm{left}} \gets \mathrm{RDP}(\tau, s, k, \epsilon)$   
13:  $\tau_{\mathrm{right}} \gets \mathrm{RDP}(\tau, k, e, \epsilon)$   
14: return  $\{p_k\} \cup \tau_{\mathrm{left}} \cup \tau_{\mathrm{right}}$   
15: else  
16: return  $\{p_s, p_e\}$   
17: end if

The key points masking adopt the Ramer-Douglas-Peucker (RDP) algorithm [10], which simplifies a trajectory by retaining points that are farthest from the line  $\overline{p_1p_n}$  connecting the first and last points. The indices are determined by

$$
\mathbf {I} _ {\text {k e y}} = \left\{p _ {k} \mid d _ {\max } \left(p _ {k}, \overline {{p _ {1} p _ {n}}}\right) > \epsilon \right\}, \tag {30}
$$

where  $\epsilon$  is a predefined threshold, and  $d_{\mathrm{max}} = \max \left\{d(p_k,\overline{p_1p_n})\mid 2\leq k\leq n - 1\right\}$  is the maximum distance measures deviation from this line. As summarized in Algorithm 1, the RDP algorithm iteratively identifies the point  $p_k$  that maximizes  $d_{\mathrm{max}} = d(p_k,\overline{p_1p_n})$ . If  $d_{\mathrm{max}} > \epsilon$ , the corresponding point  $p_k$  is treated as a key point and included in the mask set Ikey. This process is recursively applied to the trajectory segments  $\tau_{\mathrm{left}} = \{p_1,\dots ,p_k\}$  and  $\tau_{\mathrm{right}} = \{p_k,\dots ,p_n\}$ , isolating critical points

for masking. By focusing on these pivotal points, the model is challenged to reconstruct essential trajectory segments, reinforcing its understanding of key structural patterns within trajectories.

# B.2.4 Last N Masking

Last N Masking systematically removes the final N points of each trajectory:

$$
\mathbf {I} _ {\text {l a s t}} = \{n - N + 1, n - N + 2, \dots , n \}. \tag {31}
$$

This strategy explicitly simulates trajectory prediction scenarios where future positions must be forecasted based on historical observations. By incorporating this masking approach during pretraining, the model develops capabilities directly applicable to trajectory prediction tasks, creating a natural bridge between self-supervised pre-training and downstream forecasting applications.

# Corollary 3: Robustness through Comprehensive Masking

**Standpoint:** Self-supervised Trajectory Masking improves the robustness and generalization ability of the model to incomplete and heterogeneous trajectory data through a comprehensive masking strategy, enabling the model to learn more effective trajectory representations.

Proof: Let the trajectory data space be  $\mathcal{D}$ , with a true data distribution denoted as  $P(\tau)$ . In real-world applications, due to device limitations, communication failures, and environmental factors, the observed trajectories are often incomplete or irregular, and their distribution is denoted as  $P(\tilde{\tau})$ . The incompleteness of trajectory data can be formalized as a conditional distribution  $P(\tilde{\tau} \mid \tau)$ , representing the probability of observing an incomplete  $\tilde{\tau}$  given a complete trajectory  $\tau$ .

STM can be formalized as a set of masking functions  $\{\mathcal{M}_1,\mathcal{M}_2,\ldots ,\mathcal{M}_k\}$ , each corresponding to a different masking strategy. For a resampled trajectory  $\pmb {\tau}' = \{p_1,p_2,\dots ,p_n\}$ , the masking function  $\mathcal{M}_i$  transforms it as:

$$
\tilde {\boldsymbol {\tau}} _ {i} = \mathcal {M} _ {i} \left(\boldsymbol {\tau} ^ {\prime}, r _ {i}\right) = \left\{p _ {1}, \dots , \left[ \operatorname {M A S K} \right] _ {j \in \mathbf {I} _ {i}}, \dots , p _ {n} \right\} \tag {32}
$$

where  $\mathbf{I}_i\subseteq \{1,2,\dots ,n\}$  is the index set of masked positions and  $r_i = |\mathbf{I}_i| / n$  is the masking ratio.

Information-Theoretic Analysis. From an information-theoretic perspective, STM introduces an artificial information bottleneck that forces the model to learn efficient representations. We define the model objective as minimizing the reconstruction loss:

$$
\mathcal {L} (\theta) = \mathbb {E} _ {\boldsymbol {\tau} \sim P (\boldsymbol {\tau}), i \sim \mathcal {U} (1, k)} \left[ d \left(f _ {\theta} \left(\mathcal {M} _ {i} (\boldsymbol {\tau}, r _ {i})\right), \boldsymbol {\tau}\right) \right], \tag {33}
$$

where  $d$  is a chosen distance metric.

During training, the model needs to learn the joint distribution  $P(\pmb{\tau}, \tilde{\pmb{\tau}}_i)$  and estimate the conditional distribution  $P(\pmb{\tau} \mid \tilde{\pmb{\tau}}_i)$ . By Bayes' theorem:

$$
P (\boldsymbol {\tau} \mid \tilde {\boldsymbol {\tau}} _ {i}) = \frac {P \left(\tilde {\boldsymbol {\tau}} _ {i} \mid \boldsymbol {\tau}\right) P (\boldsymbol {\tau})}{P \left(\tilde {\boldsymbol {\tau}} _ {i}\right)}. \tag {34}
$$

By using diverse masking strategies, the model learns to estimate  $P(\boldsymbol{\tau} \mid \tilde{\boldsymbol{\tau}}_i)$  across different types of masked trajectories, which is equivalent to learning the true trajectory distribution  $P(\boldsymbol{\tau})$  and the various degradation mechanisms  $P(\tilde{\boldsymbol{\tau}}_i \mid \boldsymbol{\tau})$ .

Optimality Theory of Diversity Complementary Masking Strategies. A key innovation in STM is the use of multiple complementary masking strategies. We define the coverage region of the union of masking strategies as:

$$
\mathcal {C} \left(\left\{\mathcal {M} _ {1}, \dots , \mathcal {M} _ {k} \right\}\right) = \int_ {\tilde {\boldsymbol {\tau}} \in \mathcal {D}} \max  _ {i \in \{1, \dots , k \}} P _ {\mathcal {M} _ {i}} (\tilde {\boldsymbol {\tau}}) d \tilde {\boldsymbol {\tau}}, \tag {35}
$$

where  $P_{\mathcal{M}_i}(\tilde{\tau})$  denotes the distribution of incomplete trajectories generated by masking strategy  $\mathcal{M}_i$ .

We assert that for a suitable masking ratio and a diverse set of masking strategies  $\{\mathcal{M}_1, \ldots, \mathcal{M}_k\}$ , the combined coverage region satisfies:

$$
\mathcal {C} \left(\left\{\mathcal {M} _ {1}, \dots , \mathcal {M} _ {k} \right\}\right) > \max  _ {i \in \{1, \dots , k \}} \mathcal {C} \left(\left\{\mathcal {M} _ {i} \right\}\right). \tag {36}
$$

This inequality indicates that the joint use of diverse masking functions provides strictly better coverage over possible incomplete trajectories than any individual strategy.

The advantage of combining multiple masking strategies in STM over using a single masking strategy can also be theoretically justified by comparing the expected reconstruction error. Assume the real-world conditional distribution of incomplete trajectories is  $P_{\mathrm{real}}(\tilde{\tau} \mid \tau)$ . For a single masking strategy  $\mathcal{M}_i$ , let the generated distribution be  $P_{\mathcal{M}_i}(\tilde{\tau} \mid \tau)$ . Then the expected reconstruction error under this distribution is:

$$
\mathbb {E} _ {\tau \sim P (\tau), \tilde {\tau} \sim P _ {\text {r e a l}} (\tilde {\tau} | \tau)} [ d (f _ {\theta} (\tilde {\tau}), \tau) ] \tag {37}
$$

It can be shown that training with a mixture of multiple masking strategies leads to a lower bound on this error compared to using any single strategy. This is because the mixture of diverse masking strategies better approximates the true real-world distribution of incomplete trajectories:

$$
K L \left(P _ {\text {r e a l}} (\tilde {\boldsymbol {\tau}} \mid \boldsymbol {\tau}) \left\| \frac {1}{k} \sum_ {i = 1} ^ {k} P _ {\mathcal {M} _ {i}} (\tilde {\boldsymbol {\tau}} \mid \boldsymbol {\tau}) \right.\right) <   \min  _ {i \in \{1, \dots , k \}} K L \left(P _ {\text {r e a l}} (\tilde {\boldsymbol {\tau}} \mid \boldsymbol {\tau}) \| P _ {\mathcal {M} _ {i}} (\tilde {\boldsymbol {\tau}} \mid \boldsymbol {\tau})\right) \tag {38}
$$

Here,  $KL(\cdot \| \cdot)$  denotes the Kullback-Leibler divergence.

# C Details of UniTraj

In this section, we provide a detailed implementation of UniTraj, including the architecture and parameter settings.

# C.1 Overall Architecture

The UniTraj model adopts an encoder-decoder architecture based on transformer blocks, designed to process trajectory data with minimal regional dependency and maximum task adaptability. Figure 8 illustrates the overall framework of UniTraj, which consists of several key components: spatiotemporal tokenization, encoder, decoder, and rotary embedding layers.

Our model takes trajectory points that have already undergone adaptive resampling (ATR) and masking (STM) as described in Appendix B. The input trajectories are represented as sequences of latitude-longitude coordinates and timestamps:  $\tau = \{(\mathrm{lng}_i, \mathrm{lat}_i, t_i) | i = 1, 2, \dots, n\}$ , where  $n$  is the total number of points after resampling. Unlike previous approaches that rely on region-specific features or road network information, UniTraj operates solely on these basic coordinates, enhancing its universal applicability across diverse geographic contexts.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/c274d27e42cd7b90002aa5a0b2d56579c89fa25b9143135d182973dac1479f1a.jpg)



Figure 8: The main architecture and components of UniTraj.


# C.2 Input Representation and Embedding

Spatio-Temporal Tokenization. To enhance numerical stability and generalization, all input coordinates are normalized relative to the first point in the trajectory  $(x_{i},y_{i}) = (\mathrm{lng}_{i} - \mathrm{lng}_{1},\mathrm{lat}_{i} - \mathrm{lat}_{1})$

For the spatial component, we project the normalized coordinates into a  $d$ -dimensional space using a 1D convolutional neural network, yielding a spatial embedding

$$
\boldsymbol {h} _ {i} ^ {s} = \operatorname {C o n v 1 D} \left(\left[ x _ {i}, y _ {i} \right]; \theta_ {s}\right), \tag {39}
$$

where  $\theta_{s}$  represents the learnable parameters of the convolutional layer. We use a kernel size of 1 with no stride to capture local spatial dependencies. Similarly, the temporal component, based on the time intervals  $\Delta t_i$ , is embedded into the same  $d$ -dimensional space via a linear layer, resulting in a temporal embedding:

$$
\boldsymbol {h} _ {i} ^ {t} = W _ {t} \cdot \Delta t _ {i} + b _ {t}, \tag {40}
$$

where  $W_{t} \in \mathbb{R}^{d \times 1}$  and  $b_{t} \in \mathbb{R}^{d}$  are learnable parameters. The final embedding for each trajectory point is obtained by element-wise addition of the spatial and temporal components:

$$
\boldsymbol {h} _ {i} = \boldsymbol {h} _ {i} ^ {s} + \boldsymbol {h} _ {i} ^ {t} \tag {41}
$$

This dual-tokenization captures both spatial and temporal dynamics, enabling the model to learn relative movement and temporal dependencies effectively.

Rotary Positional Encoding (RoPE). In addition to encoding the spatial and temporal details of each trajectory point, it is essential to capture the relative positional relationships between points. These relationships enable the model to comprehend the movement sequence and the timing between points, both crucial for accurate trajectory modeling. To achieve this, we employ Rotary Position Encoding (RoPE) [32], which maintains the relative positional information between points by rotating the trajectory embedding vectors. Given the combined spatial-temporal embeddings  $h_i$  for point  $i$  in the trajectory, RoPE applies a rotational transformation:

$$
\operatorname {R o P E} \left(\boldsymbol {h} _ {i}\right) = \left( \begin{array}{c c} \cos \theta_ {i} & - \sin \theta_ {i} \\ \sin \theta_ {i} & \cos \theta_ {i} \end{array} \right) \left( \begin{array}{l} \boldsymbol {h} _ {i} ^ {(1)} \\ \boldsymbol {h} _ {i} ^ {(2)} \end{array} \right), \tag {42}
$$

where  $\pmb{h}_i^{(1)}$  and  $\pmb{h}_i^{(2)}$  are the first and second halves of the embedding  $\pmb{h}_i$ , and  $\theta_i$  is a rotation angle that varies proportionally with the position index  $i$ . Specifically,  $\theta_i$  is calculated as  $\theta_i = \frac{i}{10000^{2k / d}}$ , where  $k$  is the index of the embedding dimension, and  $d$  is the total dimension of the embedding.

The main advantage of RoPE is its ability to preserve relative positional information through rotational symmetry. This ensures that the relative distance and directional relationships between points are maintained, enabling the model to capture both local patterns (e.g., short-term movements) and global patterns (e.g., long-range directionality) within a trajectory. By encoding these relative positions, RoPE strengthens the model's capacity to understand movement dynamics across varying scales.

# C.3 Adaptive Representation Learning

The UniTraj employs an encoder-decoder architecture [13] tailored for trajectory data. The encoder and decoder use Transformer blocks [33] with RoPE-powered self-attention mechanisms to capture dependencies within trajectory embeddings.

Encoder. Given a masked trajectory  $\tilde{\pmb{\tau}} = \{p_1,\dots ,[\mathrm{MASK}]_{i\in \mathbf{I}},\dots ,p_n\}$ , we first extract the embedding representations of the unmasked points  $\mathbf{H} = \{h_1,h_2,\dots ,h_m\}$  (where  $m\leq n$  and  $i\notin \mathbf{I}$ ) through the tokenizer and positional encoding steps. The encoder  $\mathbf{E}_{\theta}$  processes the visible (unmasked) points in a trajectory to generate contextualized representations. It consists of  $L_{e}$  transformer blocks, each incorporating:

1. Multi-head Self-attention with RoPE: As described previously, we apply RoPE to the self-attention mechanism:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q _ {\mathrm {R o P E}} \cdot K _ {\mathrm {R o P E}} ^ {T}}{\sqrt {d _ {k}}}\right) \cdot V \tag {43}
$$

where  $Q_{\mathrm{RoPE}}$  and  $K_{\mathrm{RoPE}}$  are the query and key matrices with RoPE applied.

2. Feed-forward Network (FFN): A two-layer FFN with GELU activation:

$$
\operatorname {F F N} (x) = W _ {2} \cdot \operatorname {G E L U} \left(W _ {1} \cdot x + b _ {1}\right) + b _ {2} \tag {44}
$$

3. Layer Normalization and Residual Connections: Each sub-block is wrapped with layer normalization (Pre-LN) and residual connections:

$$
\mathbf {H} ^ {\prime} = \operatorname {L a y e r N o r m} (\mathbf {H} + \operatorname {A t t e n t i o n} (\mathbf {H})) \tag {45}
$$

$$
\mathbf {H} ^ {\prime} = \operatorname {L a y e r N o r m} \left(\mathbf {H} ^ {\prime} + \operatorname {F F N} \left(\mathbf {H} ^ {\prime}\right)\right) \tag {46}
$$

The encoder's output is a set of hidden representations  $\mathbf{H}^e = \{\pmb{h}_i^e | i = 1,2,\dots,m\}$  for the  $m$  visible points.

Decoder. The decoder reconstructs the masked points based on the contextualized representations from the encoder. It operates by combining the encoder's embeddings with mask tokens and processing them through  $L_{d}$  transformer layers:

1. Input Combination: The decoder input consists of both the encoder outputs for visible points and the mask token embeddings for masked positions:

$$
\mathbf {H} _ {0} ^ {d} = \operatorname {R e o r d e r} \left(\left\{ \begin{array}{l l} h _ {i} = \mathbf {h} _ {j} ^ {e} & \text {i f} i = \operatorname {I n d e x} (j), i \notin \mathbf {I} \\ h ^ {\text {m a s k}} & \text {i f} i \in \mathbf {I} \end{array} \right\}\right), \tag {47}
$$

where  $h^{\mathrm{mask}}$  represents the mask token embeddings for all masked positions.

2. Decoder Transformer Blocks: The combined input is processed through  $L_{d}$  transformer blocks, each with the same structure as the encoder blocks (self-attention with RoPE, FFN, layer normalization, and residual connections). The self-attention mechanism allows information to flow between visible and masked positions:

$$
\mathbf {H} _ {l} ^ {d} = \operatorname {T r a n s f o r m e r B l o c k} \left(\mathbf {H} _ {l - 1} ^ {d}\right) \tag {48}
$$

for  $l\in \{1,2,\dots ,L_d\}$

3. Output Projection: The final layer projects the decoder's representations for the masked positions back to coordinate space:

$$
\left(\hat {x} _ {j}, \hat {y} _ {j}\right) = W _ {o} \cdot \boldsymbol {h} _ {L _ {d}, j} ^ {d} + b _ {o} \tag {49}
$$

where  $j$  indexes the masked positions, and  $W_{o}\in \mathbb{R}^{2\times d}$  and  $b_{o}\in \mathbb{R}^{2}$  are learnable parameters. These projected coordinates are then transformed back to the original coordinate system  $(\hat{\mathrm{lng}}_j,\hat{\mathrm{lat}}_j) = (\hat{x}_j + \mathrm{lng}_1,\hat{y}_j + \mathrm{lat}_1)$ .

UniTraj is trained using a self-supervised learning approach with a reconstruction loss function. For each trajectory, we apply our masking strategies (random, block, key points, or last N), and the model is trained to reconstruct these masked points:

$$
\mathcal {L} = \frac {1}{| \mathbf {I} |} \sum_ {j \in \mathbf {I}} \| (\hat {x} _ {j}, \hat {y} _ {j}) - (x _ {j}, y _ {j}) \| _ {2} ^ {2}, \tag {50}
$$

where  $\mathbf{I}$  is the set of masked positions, and  $\| \cdot \|$  denotes the L2 norm.

# C.4 Task-Specific Adaptation

For downstream applications, UniTraj can be used in two primary ways:

1. Zero-shot Transfer: The pre-trained model's encoder can be directly applied to extract trajectory representations for various tasks without further training. We use the pre-trained UniTraj as a backbone and attach task-specific Multi-Layer Perceptron (MLP) adapters to the output:

$$
\mathbf {H} ^ {\text {f i n a l}} = \operatorname {M L P} \left(\operatorname {U n i T r a j} _ {\text {e n c o d e r}} (\tau)\right) \tag {51}
$$

The MLP adapter typically consists of 2-3 layers with non-linear activations:

$$
\mathbf {H} ^ {\mathrm {a d a}} = W _ {2} \cdot \operatorname {R e L U} \left(W _ {1} \cdot \mathbf {H} ^ {e} + b _ {1}\right) + b _ {2} \tag {52}
$$

where  $\mathbf{H}^e$  is the output from the UniTraj encoder.

2. Fine-tuning: Update all parameters of the backbone and adapters with specific dataset.

For different downstream tasks, we design specific adapter architectures:

- For Trajectory Recovery/Prediction: We can directly use UniTraj's decoder as an adapter without any additional modifications.

- For Trajectory Classification: The adapter includes pooling operations followed by fully connected layers to produce class logits.

- For Trajectory Generation: The adapter interfaces with generative models by providing conditioned trajectory embeddings.

# C.5 Implementation Details

Additionally, we summarize the list of key hyperparameters and implementation-specific settings that may be used in the implementation of UniTraj in Table 5. Specifically, our model contains 8 encoders and 4 decoders, each using 4 heads in the attention layer. The model has approximately 2.38 million parameters, allowing it to balance complexity and computational efficiency. We set the embedding dimension to 128 and employ RoPE to capture spatial and temporal relationships effectively. Our model can handle an arbitrary length of the number of trajectory points and pad it to a length of 200. Naturally, due to the use of rotational positional embedding, our model holds extension capability and supports a maximum length of 512. In addition, when performing the dynamic resampling strategy, we set the minimum number of sampling points to 36 and the maximum to 600, and its minimum sampling rate is 0.35. Finally, we provide the probability of using various masking strategies during training, which can be further adapted to the specific task as we discussed in Section 5.4 and Table 4.


Table 5: General parameters setting for UniTraj.


<table><tr><td>Parameter</td><td>Setting value</td><td>Refer range</td></tr><tr><td>Encoder Blocks</td><td>8</td><td>≥ 2</td></tr><tr><td>Decoder Blocks</td><td>4</td><td>≥ 2</td></tr><tr><td>Attention Heads</td><td>4</td><td>≥ 1</td></tr><tr><td>Encode Dim</td><td>128</td><td>64 ~ 256</td></tr><tr><td>Parameters of Model (Millions)</td><td>2.38</td><td>-</td></tr><tr><td>Mask ratio</td><td>0.5</td><td>0.25 ~ 0.75</td></tr><tr><td>Trajectory Length Padding</td><td>200</td><td>36 ~ 256</td></tr><tr><td>Maximum Length Padding</td><td>512</td><td>-</td></tr><tr><td>Minimum Trajectory Points</td><td>36</td><td>-</td></tr><tr><td>Maximum Trajectory Points</td><td>600</td><td>-</td></tr><tr><td>Minimum Sampling ratio</td><td>0.35</td><td>-</td></tr><tr><td>Random Masking</td><td>0.7</td><td>-</td></tr><tr><td>Key Points Masking</td><td>0.15</td><td>-</td></tr><tr><td>Block Masking</td><td>0.05</td><td>-</td></tr><tr><td>Last N Masking</td><td>0.1</td><td>-</td></tr></table>

# D Experiments Details

We use the Adam optimizer and mean square error loss with an initial learning rate of  $1 \times 10^{-3}$  with a learning rate scheduler. The model is trained for 200 epochs with a batch size of 1024, and early stopping is applied based on validation performance. All experiments were conducted using PyTorch, where the foundation model is trained on NVIDIA A100/L40s 40GB GPUs and the baseline experiments are performed on RTX 2080 Ti.

# D.1 Datasets

We evaluate the performance of the proposed model using six diverse real-world trajectory datasets. Each dataset represents different data collection scenarios, quality levels, motion patterns, and geographic regions, providing a comprehensive test of the capabilities of UniTraj.

- **WorldTrace:** WorldTrace is our proposed large-scale, globally distributed dataset, which we describe in detail in Section 4.1. We curated a high-quality subset of 1.1 million trajectories from

the original dataset, which have been filtered to remove long stops and loops. Of this subset, 1 million trajectories are designated for model training combined with resampling or masking strategies, with the remaining 100,000 reserved for testing without any operation. To ensure consistency and enable independent zero-shot evaluations, the testing dataset is normalized to a sampling interval of 3 seconds per point.

- Chengdu [8]: The Chengdu dataset comprises over one million urban mobility trajectories collected from taxis operating in Chengdu, China, reflecting daily commuting and transportation patterns in a densely urbanized area. It features dense, high-frequency (3-second for most trajectories) sampling points that provide detailed insights into active urban environments.

- Xi'an [8]: Similar to Chengdu, the Xi'an dataset includes millions of taxi trajectories gathered in Xi'an, China, focusing on movement patterns within another densely populated Chinese city. The data, collected during November 2016, captures the traffic dynamics and urban mobility behaviors specific to this region.

- GeoLife [47]: The GeoLife dataset is a widely used trajectory dataset collected over three years by 182 users, primarily in Beijing, China. It is mainly distinguished by a wide variety of travel modes, including walking, cycling and driving. With this data, we can study the trajectory movement patterns and behavioral habits of different travel modes. Besides, this dataset suffers from irregular and often long sampling intervals, which limit its granularity and quality for trajectory analysis.

- Grab-Posisi [15]: Sourced from Southeast Asia, this dataset contains 84,000 ride-hailing trajectories, predominantly from the Grab service in cities such as Jakarta and Singapore. The variable sampling intervals across these trajectories provide insights into urban mobility patterns unique to Southeast Asian metropolises.

- Porto [28]: The Porto dataset consists of taxi trajectories collected in Porto, Portugal, capturing trips between different areas of the city. Although it provides valuable insight into taxi mobility within the city, the dataset has a relatively low sampling frequency, with long intervals (15 seconds) between data points.

# D.2 Tasks Applicability Study Settings

# D.2.1 Trajectory Recovery

In this experiment, we randomly mask  $50\%$  of trajectory points and test the recovery performance. Specifically, we evaluate UniTraj in both zero-shot (trained solely on WorldTrace) and fine-tuned settings (trained on WorldTrace and then fine-tuned on each respective dataset), aiming to understand its adaptability with and without task-specific training. Additionally, we compare UniTraj against a diverse range of baselines, including traditional deep learning models (Linear, DHTR [35], Transformer [33], and DeepMove [11]) and pre-trained models (TrajBERT [31] and TrajFM [23]). Performance metrics include Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) with meters, computed based on geographic distance:

$$
\mathrm {M A E} = \frac {1}{n} \sum_ {i} ^ {n} \left| y _ {i} - \hat {y} _ {i} \right|, \tag {53}
$$

$$
\mathrm {R M S E} = \sqrt {\frac {1}{n} \sum_ {i} ^ {n} \left(y _ {i} - \hat {y} _ {i}\right) ^ {2}} \tag {54}
$$

where  $y_{i}$  and  $\hat{y}_i$  are the real and recovered coordinates, respectively.

# D.2.2 Trajectory Prediction

In this task, we focus on predicting future trajectories based on historical trajectory points. Following the setup [23] in previous work, we predicted the locations of five future points. The baseline settings and evaluation metrics are consistent with those used for the trajectory recovery task, and experiments were conducted on WorldTrace, Chengdu, and GeoLife datasets.

# D.2.3 Trajectory Classification

The Trajectory Classification task is conducted on two datasets, GeoLife and Grab-Posisi. In this task, we will only use the encoder module of the UniTraj as a backbone and then add a classification header.

We compare UniTraj in two settings: without fine-tuning (wo/ft), where only the classifier head is trained, and with fine-tuning (ft), where the entire model is updated. For baselines, we following prior literature [21] use representative classification models including GRU, LSTM, STGN [45], and TrajFormer [21]. Performance is reported by classification accuracy:

$$
\operatorname {A c c} = \frac {1}{n} \sum_ {i} ^ {n} \mathbf {I} \left(y _ {i}, \hat {y} _ {i}\right), \tag {55}
$$

where  $y_{i}$  and  $\hat{y}_i$  are the predicted and true labels, respectively, and  $\mathbf{I}(\cdot)$  is a indicator function. Following the general settings of previous work, we selected four travel modes from the Geolife dataset, namely walking, bus, bike, and driving. For the Grab-Posisi dataset, there are two travel modes: car and motorcycle.

# D.2.4 Trajectory Generation

In this task, we follow the approach in prior work [50], assessing trajectory generation using sequences of road segments that represent trajectories without explicit temporal attributes. Specifically, we use ControlTraj as a downstream task for trajectory generation, where we replace the road segment extraction component (RoadMAE) of the ControlTraj with UniTraj's encoder, testing the effectiveness of the embedded representation. The evaluation includes density error metrics [50]:

$$
\operatorname {D e n s i t y} \operatorname {E r r o r} = \operatorname {J S D} (G \| O) = \frac {1}{2} \mathbb {D} (\| \frac {(G + O)}{2}) + \frac {1}{2} \mathbb {D} (G \| \frac {(G + O)}{2}), \tag {56}
$$

where  $G$  is the distribution of the generated trajectories in the city (which divides each city into grids of  $16 \times 16$  size and calculates the count of trajectory points associated with each grid), and  $O$  is the distribution of the original trajectories. JSD(·) is the Jenson-Shannon divergence for two distributions.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/f3e268538b33c76b2e8575a5efa3204b02892a0bc9a17ab82a5a30ead332e49c.jpg)



(a) Original.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/737e429aef641ade746e335aeb30a898096ce9733d66680178579adb75451c3a.jpg)



(b) ControlTraj.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/2be045a07fb288d2d46fa70b6455ecb5d87a3b6e2c83c849838408f3b08078d0.jpg)



(c) ControlTraj + UniTraj.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/ca1244d2fbdc8cbc74416af83210e8dfbd05468f66366ad2bb71b5e7fe7ee78e.jpg)



(d) Original.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/b12f2c18d4ec3443a3e92bb840249431266f2adcbc17d6077e6ed08d22ee4f87.jpg)



(e) ControlTraj.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-01/c4632903-9014-4bd1-b265-8db98cbdce90/e4b9c29b8ec7afe2798439cbcfbcbfda5493c27cb546205e4555c08cfd1371fe.jpg)



(f) ControlTraj + UniTraj.



Figure 9: Performance comparison of trajectory generation task with Chengdu dataset (first row), and transfer to Xi'an dataset (second row).


For the this task, UniTraj demonstrates its versatility through integration with existing generative frameworks. By replacing ControlTraj's road segment extraction module with UniTraj, we achieved

a  $5.1\%$  reduction in density error (from 0.0039 to 0.0037) when trained and generated on the Chengdu dataset. This improvement, though modest in magnitude, represents a significant advance in trajectory fidelity. More impressively, when transferring the generation capability to Xi'an without retraining—a challenging cross-region scenario—the UniTraj-enhanced generator maintains a density error of 0.0152. In contrast, the baseline ControlTraj experiences a 0.0171 density error when transferred across regions. This cross-region resilience further validates UniTraj's ability to capture universal trajectory patterns that transcend specific geographic contexts. We also show the heatmap visualizations to measure the accuracy and realism of generated trajectories in Figure 9, where brighter regions indicate denser trajectories and darker regions indicate sparser ones. Detailed analysis of the generated trajectories reveals that UniTraj-enhanced generation produces more realistic speed variations, particularly in complex road segments such as intersections, sparse or dense areas. In summary, the above results underscore UniTraj's potential for robust and transferable trajectory generation, proving its effectiveness in both familiar and novel geographic settings.

# D.3 Dataset Study Settings

Effect of Dataset Scale and Quality. This task focuses on the impact of dataset size and quality on UniTraj performance. We analyze WorldTrace for the effects of different amounts and qualities of training data. Specifically, we further process the complete WorldTrace dataset by removing cyclic trajectories, removing trajectories with too many stopping points and sparse trajectories. In total, we partitioned a subset of high-quality trajectory data numbering 1 million items, and further partitioned a subset of 10,000, 500,000 trajectory data for UniTraj training.

Effect of Dataset Diversity. The task assessed the impact of using different data coverage (i.e., geographic diversity) on the model. We evaluate the zero-shot performance of UniTraj trained on the WorldTrace and Chengdu datasets, respectively, and tested on multiple real-world trajectory datasets. We chose the Chengdu dataset for comparison because it has very high data quality and has the identical me collection standards as the Xi'an dataset.

# D.4 Model Study Settings

For setting the number of encoders decoders for the model, we adopt the following scheme {encoders: 2,4,6,8,12}, {decoders:2,2,4,4,6}, {attention heads:2,2,2,4,8}. We believe that an asymmetric encoder-decoder architecture can significantly reduce the number of parameters while maximizing the performance of the model. And the scaling law between the number of model parameters and the size of the data will be one of the considerations in our future research and model architecture design.

# E More Discussion

# E.1 Limitation

While UniTraj represents a significant advancement in universal trajectory modeling, several limitations remain that warrant acknowledgment and future investigation. Despite WorldTrace's unprecedented geographic coverage spanning 70 countries, data distribution remains uneven, with certain regions (particularly in Africa and parts of Asia) underrepresented, potentially limiting model performance in these areas. Additionally, our focus on motorized movement may restrict generalization to non-motorized mobility patterns, such as pedestrian trajectories with distinctly different motion properties. The computational resources required for training and deploying UniTraj at scale present practical challenges for resource-constrained environments, necessitating more efficient architectures or distillation approaches. From a technical perspective, UniTraj relies solely on coordinate and temporal information, lacking integration of contextual features like road networks, traffic conditions, and points of interest that could further enhance predictive accuracy. Addressing these limitations represents promising directions for future research, potentially through expanded geographic coverage, multimodal trajectory integration, architecture optimization, context-aware modeling, and continual learning techniques. Nonetheless, we believe that the proposed UniTraj and WorldTrace datasets will contribute to the development of the entire community towards a more generalized, global view of trajectory analysis.

# E.2 Broader Impact

This work presents both promising opportunities and notable concerns for society. Positively, this universal trajectory model could popularize mobility intelligence across diverse regions, enabling improved transportation systems in underserved areas without extensive local data collection. The model could drive more efficient urban planning, reduce traffic congestion and emissions, and enhance logistics optimization globally. However, this technology could also enable more pervasive monitoring capabilities, raising surveillance concerns if misused. Additionally, there exists potential for widening technological disparities between resource-rich and resource-constrained organizations. Balancing these implications requires commitment to privacy-preserving techniques and equitable access policies to ensure this technology advances social welfare while minimizing potential harms.