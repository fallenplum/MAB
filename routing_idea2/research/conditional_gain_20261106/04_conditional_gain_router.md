# 条件增益路由

## 研究问题
预测同题相对训练默认组合的配对增益，是否能找出值得切换的新题。

## 实验设置与对照
默认组按训练折选，收缩伪计数10、子任务至少4题，不足回退任务/全局。同表征比较线性、默认条件交互线性和32单隐层MLP；各4个正则。阈值仅按训练OOF选择。普通线性delta重定锚与直接20回归具有严格排序等价性，作为负对照。

## 数值结果
选定配置：{"candidate": "delta_mlp__full_meta__a1", "threshold": 0.025, "oof_accuracy": 0.8766666666666667, "switch_rate": 0.8483333333333334}

{
  "conditional_main_thresholded": {
    "accuracy": 0.8616666666666668,
    "switch_questions": 188,
    "switch_rate": 0.94,
    "wrong_to_right_cells": 25,
    "right_to_wrong_cells": 30,
    "net_corrected_cells": -5,
    "changed_question_mean_positive": 15,
    "changed_question_mean_negative": 21,
    "per_generation_accuracy": [
      0.8400000000000001,
      0.865,
      0.88
    ],
    "vs_original_baselines": {
      "global_fixed": {
        "difference": 0.0016666666666666674,
        "ci95": [
          -0.02333333333333333,
          0.026666666666666665
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "task_fixed": {
        "difference": 0.0,
        "ci95": [
          -0.024999999999999998,
          0.024999999999999994
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "subtask_fixed": {
        "difference": -0.008333333333333331,
        "ci95": [
          -0.03333333333333333,
          0.01833333333333333
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "independent_top3": {
        "difference": -0.0016666666666666644,
        "ci95": [
          -0.028333333333333332,
          0.026666666666666665
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      }
    },
    "within_bbh_subtask": {
      "boolean_expressions": {
        "questions": 3,
        "switch_questions": 2,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/boolean_expressions": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "causal_judgement": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 4,
        "ci": {
          "difference": 0.3333333333333333,
          "ci95": [
            0.0,
            0.75
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/causal_judgement": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "date_understanding": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            -0.5,
            0.5
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/date_understanding": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "disambiguation_qa": {
        "questions": 4,
        "switch_questions": 3,
        "net_corrected_cells": -1,
        "ci": {
          "difference": -0.08333333333333333,
          "ci95": [
            -0.5,
            0.41666666666666663
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/disambiguation_qa": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "dyck_languages": {
        "questions": 4,
        "switch_questions": 3,
        "net_corrected_cells": -1,
        "ci": {
          "difference": -0.08333333333333333,
          "ci95": [
            -0.25,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/dyck_languages": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "formal_fallacies": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": -2,
        "ci": {
          "difference": -0.16666666666666666,
          "ci95": [
            -0.5,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/formal_fallacies": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "geometric_shapes": {
        "questions": 4,
        "switch_questions": 2,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/geometric_shapes": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "hyperbaton": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/hyperbaton": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_five_objects": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_five_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_seven_objects": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            -0.25,
            0.25
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_seven_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_three_objects": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": -1,
        "ci": {
          "difference": -0.08333333333333333,
          "ci95": [
            -0.25,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_three_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "movie_recommendation": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/movie_recommendation": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "multistep_arithmetic_two": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/multistep_arithmetic_two": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "navigate": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": -2,
        "ci": {
          "difference": -0.16666666666666666,
          "ci95": [
            -0.3333333333333333,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/navigate": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "object_counting": {
        "questions": 4,
        "switch_questions": 3,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/object_counting": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "penguins_in_a_table": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/penguins_in_a_table": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "reasoning_about_colored_objects": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/reasoning_about_colored_objects": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "ruin_names": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": -2,
        "ci": {
          "difference": -0.16666666666666666,
          "ci95": [
            -0.3333333333333333,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/ruin_names": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "salient_translation_error_detection": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": 2,
        "ci": {
          "difference": 0.2222222222222222,
          "ci95": [
            0.0,
            0.6666666666666666
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/salient_translation_error_detection": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "snarks": {
        "questions": 3,
        "switch_questions": 2,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/snarks": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "sports_understanding": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            -0.25,
            0.25
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/sports_understanding": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "temporal_sequences": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/temporal_sequences": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_five_objects": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/tracking_shuffled_objects_five_objects": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_seven_objects": {
        "questions": 4,
        "switch_questions": 2,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/tracking_shuffled_objects_seven_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_three_objects": {
        "questions": 4,
        "switch_questions": 3,
        "net_corrected_cells": -2,
        "ci": {
          "difference": -0.16666666666666666,
          "ci95": [
            -0.5,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/tracking_shuffled_objects_three_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "web_of_lies": {
        "questions": 4,
        "switch_questions": 4,
        "net_corrected_cells": -1,
        "ci": {
          "difference": -0.08333333333333333,
          "ci95": [
            -0.3333333333333333,
            0.16666666666666669
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/web_of_lies": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "word_sorting": {
        "questions": 3,
        "switch_questions": 3,
        "net_corrected_cells": -1,
        "ci": {
          "difference": -0.1111111111111111,
          "ci95": [
            -0.6666666666666666,
            0.3333333333333333
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/word_sorting": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      }
    },
    "per_task": {
      "gsm8k": {
        "questions": 100,
        "switch_questions": 98,
        "wrong_to_right": 7,
        "right_to_wrong": 5
      },
      "bbh": {
        "questions": 100,
        "switch_questions": 90,
        "wrong_to_right": 18,
        "right_to_wrong": 25
      }
    }
  },
  "empirically_calibrated_switch": {
    "accuracy": 0.8649999999999999,
    "switch_questions": 0,
    "switch_rate": 0.0,
    "wrong_to_right_cells": 0,
    "right_to_wrong_cells": 0,
    "net_corrected_cells": 0,
    "changed_question_mean_positive": 0,
    "changed_question_mean_negative": 0,
    "per_generation_accuracy": [
      0.8500000000000001,
      0.87,
      0.875
    ],
    "vs_original_baselines": {
      "global_fixed": {
        "difference": 0.004999999999999999,
        "ci95": [
          -0.013333333333333332,
          0.021666666666666667
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "task_fixed": {
        "difference": 0.003333333333333334,
        "ci95": [
          -0.013333333333333332,
          0.02
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "subtask_fixed": {
        "difference": -0.004999999999999999,
        "ci95": [
          -0.02333333333333333,
          0.015
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      },
      "independent_top3": {
        "difference": 0.0016666666666666702,
        "ci95": [
          -0.019999999999999997,
          0.024999999999999998
        ],
        "unique_questions": 200,
        "strata_support": {
          "gsm8k/": 100,
          "bbh/boolean_expressions": 3,
          "bbh/causal_judgement": 4,
          "bbh/date_understanding": 4,
          "bbh/disambiguation_qa": 4,
          "bbh/dyck_languages": 4,
          "bbh/formal_fallacies": 4,
          "bbh/geometric_shapes": 4,
          "bbh/hyperbaton": 3,
          "bbh/logical_deduction_five_objects": 4,
          "bbh/logical_deduction_seven_objects": 4,
          "bbh/logical_deduction_three_objects": 4,
          "bbh/movie_recommendation": 4,
          "bbh/multistep_arithmetic_two": 3,
          "bbh/navigate": 4,
          "bbh/object_counting": 4,
          "bbh/penguins_in_a_table": 4,
          "bbh/reasoning_about_colored_objects": 3,
          "bbh/ruin_names": 4,
          "bbh/salient_translation_error_detection": 3,
          "bbh/snarks": 3,
          "bbh/sports_understanding": 4,
          "bbh/temporal_sequences": 4,
          "bbh/tracking_shuffled_objects_five_objects": 3,
          "bbh/tracking_shuffled_objects_seven_objects": 4,
          "bbh/tracking_shuffled_objects_three_objects": 4,
          "bbh/web_of_lies": 4,
          "bbh/word_sorting": 3
        },
        "bootstrap_replicates": 2000,
        "unit": "question_cluster_with_all_generation_repeats",
        "posthoc_descriptive": true
      }
    },
    "within_bbh_subtask": {
      "boolean_expressions": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/boolean_expressions": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "causal_judgement": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/causal_judgement": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "date_understanding": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/date_understanding": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "disambiguation_qa": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/disambiguation_qa": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "dyck_languages": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/dyck_languages": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "formal_fallacies": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/formal_fallacies": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "geometric_shapes": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/geometric_shapes": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "hyperbaton": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/hyperbaton": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_five_objects": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_five_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_seven_objects": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_seven_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "logical_deduction_three_objects": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/logical_deduction_three_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "movie_recommendation": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/movie_recommendation": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "multistep_arithmetic_two": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/multistep_arithmetic_two": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "navigate": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/navigate": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "object_counting": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/object_counting": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "penguins_in_a_table": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/penguins_in_a_table": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "reasoning_about_colored_objects": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/reasoning_about_colored_objects": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "ruin_names": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/ruin_names": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "salient_translation_error_detection": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/salient_translation_error_detection": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "snarks": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/snarks": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "sports_understanding": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/sports_understanding": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "temporal_sequences": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/temporal_sequences": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_five_objects": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/tracking_shuffled_objects_five_objects": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_seven_objects": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/tracking_shuffled_objects_seven_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "tracking_shuffled_objects_three_objects": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/tracking_shuffled_objects_three_objects": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "web_of_lies": {
        "questions": 4,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 4,
          "strata_support": {
            "bbh/web_of_lies": 4
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      },
      "word_sorting": {
        "questions": 3,
        "switch_questions": 0,
        "net_corrected_cells": 0,
        "ci": {
          "difference": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "unique_questions": 3,
          "strata_support": {
            "bbh/word_sorting": 3
          },
          "bootstrap_replicates": 2000,
          "unit": "question_cluster_with_all_generation_repeats",
          "posthoc_descriptive": true
        }
      }
    },
    "per_task": {
      "gsm8k": {
        "questions": 100,
        "switch_questions": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0
      },
      "bbh": {
        "questions": 100,
        "switch_questions": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0
      }
    }
  }
}

## 统计结论
完整配对CI按题聚类，三个seed一起重采样。所有结果为开发诊断；主候选由train OOF锁定，没有按validation选赢家。

## 失败原因与边界
简单改成相对标签未必改变决策；阈值可能选择永不切换，不能将其称逐题优势。校准变体仅proper480训练、120校准，目标是已观测3seed均值误差，不承诺真实条件Delta同时界。

## 下一步
冻结由训练选择的主候选，先构建fresh必要调用并集，再按原三个seed开展一次评价；sealed200继续封存。
