# Neural Symbolic and Probabilistic Logic Papers

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of papers on Neural Symbolic and Probabilistic Logic. Papers are sorted by their uploaded dates in descending order. Each paper is with a description of a few words. Welcome to your contribution!

**\[Taxonomy\]** We devide papers into several sub-areas, including

* [**Surveys** on Neural Symbolic and Probabilistic Logic](#survey)
* [Logic-Enhanced Neural Networks (Neural Symbolic)](#logic-enhance)
  - [Modular/Concept Learning](#modular)
    - [Neural Modular Networks](#nmn)
    - [Concept Learning](#cl)
  - [Logic as Regularizer](#regularize)
* [Neural-Enhanced Symbolic Logic (Neural Symbolic)](#nn-enhance)
  - [Differential Logic](#differential)
  - [Parameterize Logic with Neural Networks](#para)
  - [Extract Logic Rules from Neural Networks](#extract)
* [Probabilistic Logic](#prob-logic)
  - [Probabilistic Logic Programming](#PLP)
  - [Markov Logic Networks](#MLN) 
* [Theoretical Papers](#theory)
* [Miscellaneous](#misc)
  - [Logic in NLP](#nlp)
  - [Logic in Reinforcement Learning](#rl)
* [Groups](#group)

## <span id='survey'>Surveys</span>
| Year | Title                  | Venue | Paper                                      | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | ------------------------------------------------------------ |
| 2022 | Neuro-Symbolic Approaches in Artificial Intelligence | National Science Review | [Paper](https://doi.org/10.1093/nsr/nwac035) | A perspective paper that provide a rough guide to key research directions, and literature pointers for anybody interested in learning more about neural-symbolic learning.|
| 2022 | A review of some techniques for inclusion of domain-knowledge into deep neural networks | Nature Scientific Reports | [Paper](https://www.nature.com/articles/s41598-021-04590-0) | Presents a survey of techniques for constructing deep networks from data and domain-knowledge. It categorises these techniques into 3 major categories: (1) changes to input representation, (2) changes to loss function, (3a) changes to model structure and (3b) changes to model parameters. |
| 2021 | Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs | AI Open | [Paper](https://arxiv.org/abs/2010.05446) | Take a thorough look at the development of the symbolic, neural and hybrid reasoning on knowledge graphs. |
| 2021 | Modular design patterns for hybrid learning and reasoning systems | arXiv | [Paper](https://arxiv.org/abs/2102.11965v1) | Analyse a large body of recent literature and we propose a set of modular design patterns for such hybrid, neuro-symbolic systems. |
| 2021 | How to Tell Deep Neural Networks What We Know | arXiv | [Paper](https://arxiv.org/abs/2107.10295) | This paper examines the inclusion of domain-knowledge by means of changes to: the input, the loss-function, and the architecture of deep networks. |
| 2020 | From Statistical Relational to Neuro-Symbolic Artificial Intelligence | IJCAI | [Paper](https://www.ijcai.org/Proceedings/2020/0688.pdf) | This survey identifies several parallels across seven different dimensions between these two fields. |
| 2020 | Symbolic Logic meets Machine Learning: A Brief Survey in Infinite Domains | SUM | [Paper](https://arxiv.org/abs/2006.08480) | Survey work that provides further evidence for the connections between logic and learning. |
| 2020 | Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective | IJCAI | [Paper](https://www.ijcai.org/Proceedings/2020/0679.pdf) | A Survey on Neural-Symbolic with GNN. |
| 2020 | Symbolic, Distributed and Distributional Representations for Natural Language Processing in the Era of Deep Learning: a Survey | Frontiers in Robotics and AI | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7805717/) | In this paper we make a survey that aims to renew the link between symbolic representations and distributed/distributional representations. |
| 2020 | On the Binding Problem in Artificial Neural Networks | arXiv | [Paper](https://arxiv.org/abs/2012.05208) | In this paper, we argue that the underlying cause for this shortcoming is their inability to dynamically and flexibly bind information that is distributed throughout the network. |
| 2019 | Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning | Journal of Applied Logic | [Paper](https://arxiv.org/pdf/1905.06088.pdf) | We survey recent accomplishments of neural-symbolic computing as a principled methodology for integrated machine learning and reasoning. | 
| 2017 | Neural-Symbolic Learning and Reasoning: A Survey and Interpretation | arXiv | [Paper](https://arxiv.org/pdf/1711.03902.pdf) | Reviews personal ideas and views of several researchers on neural-symbolic learning and reasoning. |
| 2011 | Statistical Relational AI: Logic, Probability and Computation | ICLP | [Paper](https://www.cs.ubc.ca/~poole/papers/SRAI-2011.pdf) | We overview the foundations of StarAI. |

## <span id='logic-enhance'>Logic-Enhanced Neural Networks</span>
### <span id='modular'>Modular/Concept Learning (Visual Question Answering)</span>
#### <span id ='nmn'>Neural Modular Networks</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Meta Module Network for Compositional Visual Reasoning | WACV | [Paper](https://arxiv.org/pdf/1910.03230) | [Code](https://github.com/wenhuchen/Meta-Module-Network) | N2NMN application|
| 2020 | Neural Module Networks for Reasoning over Text | ICLR | [Paper](https://arxiv.org/pdf/1912.04971) | [Code](https://github.com/nitishgupta/nmn-drop) | **TMN**, parser-NMN application|
| 2020 | Learning to Discretely Compose Reasoning Module Networks for Video Captioning | arXiv | [Paper](https://arxiv.org/pdf/2007.09049) | [Code](https://github.com/tgc1997/RMN) | **RMN**, N2NMN application|
| 2020 | LRTA: A Transparent Neural-symbolic Reasoning Framework with Modular Supervision for VQA | arXiv | [Paper](https://arxiv.org/pdf/2011.10731) || N2NMN application|
| 2019 | Self-Assembling Modular Networks for Interpretable Multi-hop Reasoning | arXiv | [Paper](https://arxiv.org/abs/1909.05803) | [Code](https://github.com/jiangycTarheel/NMN-MultiHopQA) | N2NMN application|
| 2019 | Probabilistic Neural-Symbolic Models for Interpretable Visual Question Answering | ICML  | [Paper](https://arxiv.org/abs/1902.07864) | [Code](https://github.com/kdexd/probnmn-clevr) | The author proposed **ProbNMN**, using variational method to generate reasoning graph. |
| 2019 | Explainable and Explicit Visual Reasoning over Scene Graphs | CVPR | [Paper](https://arxiv.org/pdf/1812.01855) | [Code](https://github.com/shijx12/XNM-Net) | **XNM**, N2NMN + scene graph|
| 2019 | Learning to Assemble Neural Module Tree Networks for Visual Grounding | ICCV | [Paper](https://arxiv.org/pdf/1812.03299.pdf) | [Code](https://github.com/daqingliu/NMTree) | **NMTree**, parser-NMN application|
| 2019 | Structure Learning for Neural Module Networks | EACL | [Paper](https://arxiv.org/pdf/1905.11532.pdf) || **LNMN**, follows Stack-NMN to add learnable (soft) modules|
| 2018 | Explainable Neural Computation via Stack Neural Module Networks | ECCV |[Paper](https://arxiv.org/pdf/1807.08556) | [Code](https://github.com/ronghanghu/snmn) | **Stack-NMN**, N2NMN + differentiable memory stack + soft program execution |
| 2018 | Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding | arXiv | [Paper](https://arxiv.org/pdf/1810.02338) | [Code](https://github.com/kexinyi/ns-vqa) | **NS-VQA**, N2NMN + scene graph|
| 2018 | Compositional Models for VQA: Can Neural Module Networks Really Count? | BICA | [Paper](https://www.sciencedirect.com/science/article/pii/S1877050918323986) || interesting (negative) result of N2NMN |
| 2018 | Transparency by Design: Closing the Gap between Performance and Interpretability in Visual Reasoning | CVPR | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mascharka_Transparency_by_Design_CVPR_2018_paper.pdf) | [Code](https://github.com/davidmascharka/tbd-nets) | **TbD**, soft modules / structures|
| 2018 | Visual Question Reasoning on General Dependency Tree | CVPR | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Visual_Question_Reasoning_CVPR_2018_paper.pdf) |[Code](https://github.com/bezorro/ACMN-Pytorch) | **ACMN**, parser-NMN (DPT -> structure)|
| 2017 | Learning to Reason: End-To-End Module Networks for Visual Question Answering | ICCV | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hu_Learning_to_Reason_ICCV_2017_paper.pdf) |[Code](https://github.com/ronghanghu/n2nmn) | **N2NMN**|
| 2017 | Inferring and Executing Programs for Visual Reasoning | ICCV | [Paper](https://arxiv.org/pdf/1705.03633) | [Code](https://github.com/facebookresearch/clevr-iep)| Basically N2NMN which refers N2NMN as "concurrent work" |
| 2016 | Learning to Compose Neural Networks for Question Answering | NAACL  | [Paper](https://arxiv.org/abs/1601.01705) | [Code](https://github.com/jacobandreas/nmn2) | Compared to original NMN, the authors add a layout selector to select layout from several proposed candidates. |
| 2016 | Neural Module Networks | CVPR  | [Paper](https://arxiv.org/abs/1511.02799v4) | [Code](https://github.com/jacobandreas/nmn2) | Initial paper. The authors proposed Neural Module Networks in this paper. |
#### <span id='cl'>Concept Learning</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Calibrating Concepts and Operations: Towards Symbolic Reasoning on Real Images | ICCV | [Paper](https://arxiv.org/pdf/2110.00519.pdf) | [Code](https://github.com/Lizw14/CaliCO) | we introduce an executor with learnable concept embedding magnitudes for handling distribution imbalance, and an operation calibrator for highlighting important operations and suppressing redundant ones | 
| 2019 | The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision | ICLR | [Paper](https://arxiv.org/pdf/1904.12584.pdf) | [Code](https://github.com/vacancy/NSCL-PyTorch-Release) | Neuro-Symbolic Concept Learner in VQA |
| 2017 | β-VAE: Learning Basiz Visual Concept With A Constrained Variational Framework | ICLR | [Paper](https://openreview.net/pdf?id=Sy2fzU9gl) | | Automated discovery of interpretable factorised latent representations from raw image |
#### Others
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2020 | Neuro-Symbolic Visual Reasoning: Disentangling "Visual" from "Reasoning" | PMLR | [Paper](http://proceedings.mlr.press/v119/amizadeh20a.html) | [Code](https://github.com/microsoft/DFOL-VQA) | a Differentiable First-Order Logic formalism for VQA |
| 2019 | Learning by Abstraction: The Neural State Machine | NeurIPS | [Paper](https://arxiv.org/abs/1907.03950) | | Given an image, we first predict a probabilistic graph then perform sequential reasoning over the graph. |

### <span id='regularize'>Logic as Regularizer</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2020 | A Constraint-Based Approach to Learning and Explanation | AAAI | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5774) | [Code](https://github.com/gabrieleciravegna/Information-based-Constrained-Learning) | Learning First Order Constraints |
| 2018 | A Semantic Loss Function for Deep Learning with Symbolic Knowledge | ICML | [Paper](https://arxiv.org/pdf/1711.11157.pdf) | [Code](https://github.com/UCLA-StarAI/Semantic-Loss/) | Semantic Loss, a continuous regularizer of logic prior. |
| 2017 | Logic tensor networks for semantic image interpretation. | IJCAI | [Paper](https://arxiv.org/abs/1705.08968) | [Code](https://gitlab.fbk.eu/donadello/LTN_IJCAI17) | Logic Tensor Networks (LTNs) are an SRL framework which integrates neural networks with first-order fuzzy logic. |
| 2017 | Semantic-based regularization for learning and inference | Artificial Intelligence | [Paper](https://www.sciencedirect.com/science/article/pii/S0004370215001344) | | A Regularizer using fuzzy logic. |
| 2016 | Harnessing Deep Neural Networks with Logic Rules | ACL | [Paper](https://arxiv.org/pdf/1603.06318.pdf) | | We propose a general framework capable of enhancing various types of neural networks (e.g., CNNs and RNNs) with declarative first-order logic rules. |

### <span id='extract'>Extract Logic Rules from Neural Networks</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Acquisition of Chess Knowledge in AlphaZero | arXiv| [Paper](https://arxiv.org/pdf/2111.09259.pdf) | | In this work we provide evidence that human knowledge is acquired by the AlphaZero neural network as it trains on the game of chess. |
| 2021 | Knowledge Neurons in Pretrained Transformers | arXiv | [Paper](https://arxiv.org/pdf/2104.08696.pdf) | | We explore how implicit knowledge is stored in pretrained Transformers by introducing the concept of *knowledge neurons*. |
| 2019 | Logical Explanations for Deep Relational Machines Using Relevance Information | JMLR | [Paper](https://www.jmlr.org/papers/v20/18-517.html) | | This work provides a methodology to generate symbolic explanations for predictions made by a deep neural network constructed from relational data, called [DRMs](https://link.springer.com/chapter/10.1007/978-3-319-99960-9_2). It investigates the use of a Bayes-like approach to identify logical proxies for local predictions of a DRM. |

## <span id='nn-enhance'>Neural-Enhanced Symbolic Logic & Deep Logic</span>
### <span id='differential'>Differential Logic</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Inclusion of domain-knowledge into GNNs using mode-directed inverse entailment | Machine Learning Journal | [Paper](https://link.springer.com/article/10.1007%2Fs10994-021-06090-8) | [Code](https://github.com/tirtharajdash/BotGNN) | Constructing GNNs from relational data and symbolic domain-knowledge, via construction of "Bottom-Graphs" |
| 2021 | Incorporating symbolic domain knowledge into graph neural networks | Machine Learning Journal | [Paper](https://link.springer.com/article/10.1007/s10994-021-05966-z) | [Code](https://github.com/tirtharajdash/VEGNN) | Constructing GNNs from relational data and symbolic domain-knowledge, via "Vertex Enrichment" |
| 2020 | Logical Neural Networks | NeurIPS | [Paper](https://arxiv.org/abs/2006.13155) | | Transform a logic formula to NN-like. Relax Boolean to \[0,1\] | 
| 2019 | Synthesizing datalog programs using numerical relaxation. | IJCAI | [Paper](https://arxiv.org/abs/1906.00163) | [Code](https://github.com/petablox/difflog) | Differential Datalog |
| 2019 | SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver | ICML | [Paper](http://proceedings.mlr.press/v97/wang19e/wang19e.pdf) | [Code](https://github.com/locuslab/SATNet) | Differential SAT |
| 2018 | Large-Scale Assessment of Deep Relational Machines | ILP | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-99960-9_2) | | Constructs MLPs from relational data and symbolic domain-knowledge using "Propositionalisation" |
| 2018 | Lifted Relational Neural Networks: Efficient Learning of Latent Relational Structures | JAIR | [Paper](https://doi.org/10.1613/jair.1.11203) | [Code](https://github.com/GustikS/NeuraLogic) | Creating deep neural networks from "templates" constructed from first-order logic rules. |
| 2018 | Learning Explanatory Rules from Noisy Data | JAIR | [Paper](https://arxiv.org/abs/1711.04574v2) | [Code](https://github.com/ai-systems/DILP-Core) | Differentiable ILP |
| 2017 | TensorLog: Deep Learning Meets Probabilistic Databases | arXiv | [Paper](https://arxiv.org/pdf/1707.05390.pdf) | [Code](https://github.com/TeamCohen/TensorLog) | Relax Boolean truth value to \[0,1\] |
| 2017 | Differentiable Learning of Logical Rules for Knowledge Base Reasoning | NeurIPS | [Paper](https://proceedings.neurips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf) | [Code](https://github.com/fanyangxyz/Neural-LP) | Neural Logic Programming, learning probabilistic first-order logical rules for knowledge base reasoning in end-to-end model. |
| 2017 | End-to-end Differentiable Proving | NeurIPS | [Paper](https://arxiv.org/abs/1705.11040) | [Code](https://github.com/uclnlp/ntp) | We replace symbolic unification with a differentiable computation on vector representations of symbols using a radial basis function kernel. |

### <span id='para'>Parameterize Logic with Neural Networks</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Neural Markov Logic Networks | UAI | [Paper](https://arxiv.org/abs/1905.13462) | [Code](https://github.com/GiuseppeMarra/nmln/tree/uai2021) | NMLNs are an exponential-family model for modelling distributions over possible worlds without explicit logic rules. |
| 2020 | NeurASP: Embracing Neural Networks into Answer Set Programming | IJCAI | [Paper](https://www.ijcai.org/Proceedings/2020/243) | [Code](https://github.com/azreasoners/NeurASP) | NeurASP, a simple extension of answer set programs by embracing neural networks. |
| 2019 | Neural Logic Machines | ICLR | [Paper](https://arxiv.org/abs/1904.11694) | [Code](https://github.com/google/neural-logic-machines) | Logic predicates as tensors, logic rules as neural operators. |
| 2019 | DeepLogic: Towards End-to-End Differentiable Logical Reasoning | AAAI-MAKE | [Paper](https://arxiv.org/abs/1805.07433) | [Code](https://github.com/nuric/deeplogic) | Feed logic rules into RNN as a string |
| 2018 | DeepProbLog: Neural Probabilistic Logic Programming | NeurIPS | [Paper](https://papers.nips.cc/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf) | [Code](https://github.com/ML-KULeuven/deepproblog) | Add "neural predicates" to ProbLog which is a probabilistic logic programming language. |

### Others
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Neural-Symbolic Integration: A Compositional Perspective | AAAI | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-10094.TsamouraE.pdf) |  | Treating Neural and Symbolic as black boxes to be integrated, without making assumptions on their internal structure and semantics. |
| 2020 | Relational Neural Machines | ECAI | [Paper](https://arxiv.org/abs/2002.02193) | | Relational Neural Machines, a novel framework allowing to jointly train the parameters of the learners and of a First–Order Logic based reasoner. |
| 2020 | Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning | ICML | [Paper](https://liqing-ustc.github.io/NGS/) | [Code](https://github.com/liqing-ustc/NGS) | NGS, (1) introducing the grammar model as a symbolic prior, (2) proposing a novel back-search algorithm to propagate the error through the symbolic reasoning module efficiently. |
| 2019 | NLProlog: Reasoning with Weak Unification for Question Answering in Natural Language | ACL | [Paper](https://arxiv.org/abs/1906.06187) |  | A Prolog prover which we extend to utilize a similarity function over pretrained sentence encoders. |
| 2018 | Lifted relational neural networks: Efficient learning of latent relational structures. | JAIR | [Paper](http://oucsace.cs.ohio.edu/~chelberg/classes/680/paperPresentations/lifted_relational_neural_networks.pdf) | | Combine the interpretability and expressive power of first order logic with the effectiveness of neural network learning. |

## <span id='prob-logic'>Probabilistic Logic</span>
### <span id='PLP'>Probabilistic Logic Programming</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2007 | ProbLog: A Probabilistic Prolog and its Application in Link Discovery | IJCAI | [Paper](https://www.ijcai.org/Proceedings/07/Papers/397.pdf) | [Code](https://github.com/ML-KULeuven/problog) | ProbLog, a library for probabilistic logic programming. |
| 2005 | Learning the structure of Markov logic networks | ICML | [Paper](https://dl.acm.org/doi/abs/10.1145/1102351.1102407) | | an algorithm for learning the structure of MLNs from relational databases |
| 2001 | Bayesian Logic Programs | | [Paper](https://arxiv.org/pdf/cs/0111058.pdf) | | Bayesian networks + Logic Program |
| 2001 | Parameter Learning of Logic Programs for Symbolic-statistical Modeling | JAIR | [Paper](https://www.jair.org/index.php/jair/article/view/10291/24551) | | Wepropose a logical/mathematical framework for statistical parameter learning of parameterized logic programs, i.e. definite clause programs containing probabilistic facts with a parameterized distribution. |
| 1996 | Stochastic Logic Programs | Advances in ILP | Paper | | A formulaton: Stochastic Logic Programs |
| 1992 | Probabilistic logic programming | Information and Computation | [Paper](https://www.sciencedirect.com/science/article/pii/089054019290061J) | | A formulation of Probabilistic logic programming. |
### <span id='MLN'>Markov Logic Networks</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2008 | Event Modeling and Recognition Using Markov Logic Networks | ECCV | [Paper](https://link.springer.com/chapter/10.1007/978-3-540-88688-4_45) | | Application of MLNs | 
| 2008 | Hybrid Markov Logic Networks | AAAI | [Paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-175.pdf) | | Extend Markov Logic Networks to continous space. |
| 2007 | Efficient Weight Learning for Markov Logic Networks | PKDD | [Paper](https://link.springer.com/chapter/10.1007/978-3-540-74976-9_21) | | weights learning of MLNs |
| 2005 | Discriminative Training of Markov Logic Networks | AAAI | [Paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-175.pdf) | | a discriminative approach to training MLNs |
| 2005 | Markov Logic Networks | Springer | [Paper](https://link.springer.com/content/pdf/10.1007/s10994-006-5833-1.pdf) | | Combining Logic and Markov Networks, a classic paper. |

## <span id='theory'>Theory</span>
| Year | Title                  | Venue | Paper                                      | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | ------------------------------------------------------------ |
| 2019 | Logical Explanations for Deep Relational Machines Using Relevance Information | JMLR | [Paper](https://www.jmlr.org/papers/volume20/18-517/18-517.pdf) | Our interest in this paper is in the construction of symbolic explanations for predictions made by a deep neural network on DRM |
| 2018 | Exact Learning of Lightweight Description Logic Ontologies | JMLR | [Paper](https://www.jmlr.org/papers/volume18/16-256/16-256.pdf) | We study the problem of learning description logic (DL) ontologies in Angluin et al.’s framework of exact learning via queries. |
| 2017 | Hinge-Loss Markov Random Fields and Probabilistic Soft Logic | JMLR | [Paper](https://www.jmlr.org/papers/volume18/15-631/15-631.pdf) | In this paper, we introduce two new formalisms for modeling structured data, and show that they can both capture rich structure and scale to big data. |
| 2017 | Answering FAQs in CSPs, Probabilistic Graphical Models, Databases, Logic and Matrix Operations (Invited Talk) | STOC | [Paper](https://dl.acm.org/doi/pdf/10.1145/3055399.3079073) | A invited talk on a general framework |

## <span id='misc'>Miscellaneous</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2020 | Integrating Logical Rules Into Neural Multi-Hop Reasoning for Drug Repurposing | ICML | [Paper](https://arxiv.org/abs/2007.05292) | | Logic Rules + GNN + RL |
| 2020 | WHAT CAN NEURAL NETWORKS REASON ABOUT? | ICLR | [Paper](https://arxiv.org/pdf/1905.13211.pdf) | | How NN structured correlates with the performance on different reasoning tasks. |
| 2019 | Bridging Machine Learning and Logical Reasoning by Abductive Learning | NeurIPS | [Paper](http://www.lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/neurips19abl.pdf) | | machine learning model learns to perceive primitive logic facts from data, while logical reasoning can exploit symbolic domain knowledge and correct the wrongly perceived facts for improving the machine learning models |
| 2013 | Deep relational machines | NeurIPS | [Paper](https://link.springer.com/chapter/10.1007/978-3-642-42042-9_27) | | A DRM learns the first layer of representation by inducing first order Horn clauses and the successive layers are generated by utilizing restricted Boltzmann machines. |


### <span id='rl'>Logic in Reinforcement Learning</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Off-Policy Differentiable Logic Reinforcement Learning | ECML PKDD | [Paper](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_49.pdf) | | In this paper, we proposed an Off-Policy Differentiable Logic Reinforcement Learning (OPDLRL) framework to inherit the benefits of interpretability and generalization ability in Differentiable Inductive Logic |
| 2020 | Exploring Logic Optimizations with Reinforcement Learning and Graph Convolutional Network | MLCAD | [Paper](https://baloneymath.github.io/files/MLCAD20_ls.pdf) | [Code](https://github.com/krzhu/abcRL) | We propose a Markov decision process (MDP) formulation of the logic synthesis problem and a reinforcement learning (RL) algorithm incorporating with graph convolutional network to explore the solution search space. |
| 2020 | Reinforcement Learning with External Knowledge by using Logical Neural Networks | IJCAI Workshop | [Paper](https://arxiv.org/abs/2103.02363) |  | We propose an integrated method that enables model-free reinforcement learning from external knowledge sources in an LNNs-based logical constrained framework such as action shielding and guide. |
| 2019 | Transfer of Temporal Logic Formulas in Reinforcement Learning | IJCAI | [Paper](https://www.ijcai.org/Proceedings/2019/0557.pdf) |  | We first propose an inference technique to extract metric interval temporal logic (MITL) formulas in sequential disjunctive normal form from labeled trajectories collected in RL of the two tasks. |
| 2019 | Neural Logic Reinforcement Learning | ICML | [Paper](https://arxiv.org/abs/1904.10729) | [Code](https://arxiv.org/abs/1904.10729) | We propose a novel algorithm named Neural Logic Reinforcement Learning (NLRL) to represent the policies in reinforcement learning by first-order logic. |

### <span id='nlp'>Natural Language Question Answering</span>
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2020 | Measuring Compositional Generalization: A Comprehensive Method on Realistic Data | ICLR | [Paper](https://arxiv.org/abs/1912.09713) | [Code](https://github.com/google-research/google-research/tree/master/cfq) | CFQ, a large dataset of Natural Language Question Answering |


## Platforms
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Domiknows: A library for integration of symbolic domain knowledge in deep learning | arXiv | [Homepage](https://hlr.github.io/domiknows-nlp/) | [Code](https://github.com/HLR/DomiKnowS) | This library provides a language interface integrate Domain Knowldge in Deep Learning. |
| 2019 | LYRICS: a General Interface Layer to Integrate Logic Inference and Deep Learning | ECML | [Paper](https://arxiv.org/abs/1903.07534) || Tensorflow, seems only in design, not implemented |
| 2007 | ProbLog: A Probabilistic Prolog and its Application in Link Discovery | IJCAI | [Paper](https://www.ijcai.org/Proceedings/07/Papers/397.pdf) | [Code](https://github.com/ML-KULeuven/problog) | ProbLog, a library for probabilistic logic programming. |
<!-- 
## Tasks
* Visual Question Answering
* Knowledge Graph Reasoning
* Question Answering in Natural Language: MedHop, WikiHop -->
<!-- 
## <span id='theory'>Groups</span>
* [Tsinghua SAIL Group](https://ml.cs.tsinghua.edu.cn/)
* [UCLA StarAI Lab](http://starai.cs.ucla.edu/)
* [Luc De Raedt @ KU Leuven](https://wms.cs.kuleuven.be/people/lucderaedt/)
* [Jian Tang @ Mila](https://tangjianpku.github.io/)
* [Tirtharaj Dash @ BITS](https://tirtharajdash.github.io/) -->
