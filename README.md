# Neural Symbolic and Probabilistic Logic Papers

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of papers on Neural Symbolic and Probabilistic Logic. Inspired by [this repo](https://github.com/safe-graph/graph-adversarial-learning-literature). Papers are sorted by their uploaded dates in descending order. Each paper is with a description of a few words. Looking forward to your contribution!

**\[Taxonomy\]** We devide papers into several sub-areas, including

* **Surveys** on Neural Symbolic and Probabilistic Logic
* Logic-Enhanced Neural Networks (Neural Symbolic)
  - Neural Modular Networks
  - Logic as Regularizer
* Neural-Enhanced Symbolic Logic (Neural Symbolic)
  - Neural Logic Programming
* Logic-Enhanced Probability (Probabilistic Logic)
* Probablity-Enhanced Logic (Probabilistic Logic)
* Theoretical Papers
* Tasks

## Surveys
| Year | Title                  | Venue | Paper                                      | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | ------------------------------------------------------------ |
| 2021 | Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs | AI Open | [Link](https://arxiv.org/abs/2010.05446) | Take a thorough look at the development of the symbolic, neural and hybrid reasoning on knowledge graphs. |
| 2021 | Modular design patterns for hybrid learning and reasoning systems | arXiv | [Link](https://arxiv.org/abs/2102.11965v1) | Analyse a large body of recent literature and we propose a set of modular design patterns for such hybrid, neuro-symbolic systems. |
| 2021 | How to Tell Deep Neural Networks What We Know | arXiv | [Link](https://arxiv.org/abs/2107.10295) | This paper examines the inclusion of domain-knowledge by means of changes to: the input, the loss-function, and the architecture of deep networks. |
| 2020 | From Statistical Relational to Neuro-Symbolic Artificial Intelligence | IJCAI | [Link](https://www.ijcai.org/Proceedings/2020/0688.pdf) | This survey identifies several parallels across seven different dimensions between these two fields. |
| 2020 | Symbolic Logic meets Machine Learning: A Brief Survey in Infinite Domains | SUM | [Link](https://arxiv.org/abs/2006.08480) | Survey work that provides further evidence for the connections between logic and learning. |
| 2020 | Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective | IJCAI | [Link](https://www.ijcai.org/Proceedings/2020/0679.pdf) | A Survey on Neural-Symbolic with GNN. |
| 2020 | Symbolic, Distributed and Distributional Representations for Natural Language Processing in the Era of Deep Learning: a Survey | Frontiers in Robotics and AI | [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7805717/) | In this paper we make a survey that aims to renew the link between symbolic representations and distributed/distributional representations. |
| 2017 | Neural-Symbolic Learning and Reasoning: A Survey and Interpretation | arXiv | [Link](https://arxiv.org/pdf/1711.03902.pdf) | Reviews personal ideas and views of several researchers on neural-symbolic learning and reasoning. |
| 2005 | Dimensions of Neural-symbolic Integration — A Structured Survey | arXiv | [Link](https://arxiv.org/abs/cs/0511042) | This work presents a comprehensive survey of the field of neural-symbolic integration. |

## Neural Modular Networks Papers
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Meta Module Network for Compositional Visual Reasoning | WACV ||| N2NMN application|
| 2020 | Neural Module Networks for Reasoning over Text | ICLR ||| **TMN**, parser-NMN application|
| 2020 | Learning to Discretely Compose Reasoning Module Networks for Video Captioning | arXiv ||| **RMN**, N2NMN application|
| 2020 | LRTA: A Transparent Neural-symbolic Reasoning Framework with Modular Supervision for VQA | arXiv ||| N2NMN application|
| 2019 | Self-Assembling Modular Networks for Interpretable Multi-hop Reasoning | arXiv | [Link](https://arxiv.org/abs/1909.05803) | [Link](https://github.com/jiangycTarheel/NMN-MultiHopQA) | N2NMN application|
| 2019 | Probabilistic Neural-Symbolic Models for Interpretable Visual Question Answering | ICML  | [Link](https://arxiv.org/abs/1902.07864) | [Link](https://github.com/kdexd/probnmn-clevr) | The author proposed **ProbNMN**, using variational method to generate reasoning graph. |
| 2019 | Explainable and Explicit Visual Reasoning over Scene Graphs | CVPR ||| **XNM**, N2NMN + scene graph|
| 2019 | Learning to Assemble Neural Module Tree Networks for Visual Grounding | ICCV ||| **NMTree**, parser-NMN application|
| 2019 | Structure Learning for Neural Module Networks | EACL ||| **LNMN**, follows Stack-NMN to add learnable (soft) modules|
| 2018 | Explainable Neural Computation via Stack Neural Module Networks | ECCV ||| **Stack-NMN**, N2NMN + differentiable memory stack + soft program execution |
| 2018 | Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding | arXiv ||| **NS-VQA**, N2NMN + scene graph|
| 2018 | Compositional Models for VQA: Can Neural Module Networks Really Count? | BICA | [Link](https://www.sciencedirect.com/science/article/pii/S1877050918323986) || interesting (negative) result of N2NMN |
| 2018 | Transparency by Design: Closing the Gap between Performance and Interpretability in Visual Reasoning | CVPR | [Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mascharka_Transparency_by_Design_CVPR_2018_paper.pdf) || **TbD**, soft modules / structures|
| 2018 | Visual Question Reasoning on General Dependency Tree | CVPR | [Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Visual_Question_Reasoning_CVPR_2018_paper.pdf) || **ACMN**, parser-NMN (DPT -> structure)|
| 2017 | Learning to Reason: End-To-End Module Networks for Visual Question Answering | ICCV | [Link](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hu_Learning_to_Reason_ICCV_2017_paper.pdf) || **N2NMN**|
| 2017 | Inferring and Executing Programs for Visual Reasoning | ICCV ||| Basically N2NMN which refers N2NMN as "concurrent work" |
| 2016 | Learning to Compose Neural Networks for Question Answering | NAACL  | [Link](https://arxiv.org/abs/1601.01705) | [Link](https://github.com/jacobandreas/nmn2) | Compared to original NMN, the authors add a layout selector to select layout from several proposed candidates. |
| 2016 | Neural Module Networks | CVPR  | [Link](https://arxiv.org/abs/1511.02799v4) | [Link](https://github.com/jacobandreas/nmn2) | Initial paper. The authors proposed Neural Module Networks in this paper. |

## Neural Logic
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Neural-Symbolic Integration: A Compositional Perspective | AAAI | [Paper](https://www.aaai.org/AAAI21Papers/AAAI-10094.TsamouraE.pdf) |  | Treating Neural and Symbolic as black boxes to be integrated, without making assumptions on their internal structure and semantics. |
| 2020 | Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning | ICML | [Paper](https://liqing-ustc.github.io/NGS/) | [Code](https://github.com/liqing-ustc/NGS) | NGS, (1) introducing the grammar model as a symbolic prior, (2) proposing a novel back-search algorithm to propagate the error through the symbolic reasoning module efficiently. |
| 2019 | NLProlog: Reasoning with Weak Unification for Question Answering in Natural Language | ACL | [Paper](https://arxiv.org/abs/1906.06187) |  | A Prolog prover which we extend to utilize a similarity function over pretrained sentence encoders. |
| 2018 | Lifted relational neural networks: Efficient learning of latent relational structures. | JAIR | [Paper](http://oucsace.cs.ohio.edu/~chelberg/classes/680/paperPresentations/lifted_relational_neural_networks.pdf) | | Combine the interpretability and expressive power of first order logic with the effectiveness of neural network learning. |
| 2017 | Differentiable Learning of Logical Rules for Knowledge Base Reasoning | NeurIPS | [Paper](https://proceedings.neurips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf) | [Code](https://github.com/fanyangxyz/Neural-LP) | Neural Logic Programming, learning probabilistic first-order logical rules for knowledge base reasoning in end-to-end model. |
| 2017 | End-to-end Differentiable Proving | NeurIPS | [Paper](https://arxiv.org/abs/1705.11040) | [Code](https://github.com/uclnlp/ntp) | We replace symbolic unification
with a differentiable computation on vector representations of symbols using a radial basis function kernel. |

### Differential Logic
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2019 | Synthesizing datalog programs using numerical relaxation. | IJCAI | [Paper](https://arxiv.org/abs/1906.00163) | [Code](https://github.com/petablox/difflog) | Differential Datalog |
| 2018 | Learning Explanatory Rules from Noisy Data | JAIR | [Paper](https://arxiv.org/abs/1711.04574v2) | [Code](https://github.com/ai-systems/DILP-Core) | Differentiable ILP |

### Parameterize Logic with Neural Networks
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2021 | Neural Markov Logic Networks | UAI | [Paper](https://arxiv.org/abs/1905.13462) | [Code](https://github.com/GiuseppeMarra/nmln/tree/uai2021) | NMLNs are an exponential-family model for modelling distributions over possible worlds without explicit logic rules. |
| 2020 | NeurASP: Embracing Neural Networks into Answer Set Programming | IJCAI | [Paper](https://www.ijcai.org/Proceedings/2020/243) | [Code](https://github.com/azreasoners/NeurASP) | NeurASP, a simple extension of answer set programs by embracing neural networks. |
| 2019 | Neural Logic Machines | ICLR | [Paper](https://arxiv.org/abs/1904.11694) | [Code](https://github.com/google/neural-logic-machines) | Logic predicates as tensors, logic rules as neural operators. |
| 2018 | DeepProbLog: Neural Probabilistic Logic Programming | NeurIPS | [Paper](https://papers.nips.cc/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf) | [Code](https://github.com/ML-KULeuven/deepproblog) | Add "neural predicates" to ProbLog which is a probabilistic logic programming language. |

### Logic as Regularizer
| Year | Title                  | Venue | Paper                                      | Code     | Description                                                  |
| ---- | ---------------------- | ----- | ------------------------------------------ | -------- | ------------------------------------------------------------ |
| 2017 | Logic tensor networks for semantic image interpretation. | IJCAI | [Paper](https://arxiv.org/abs/1705.08968) | [Code](https://gitlab.fbk.eu/donadello/LTN_IJCAI17) | Logic Tensor Networks (LTNs) are an SRL framework which integrates neural networks with first-order fuzzy logic. |

# Tasks
* Visual Question Answering
* Knowledge Graph Reasoning
* Question Answering in Natural Language: MedHop, WikiHop
