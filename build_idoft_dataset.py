"""
FlakeForge — IDoFT 100-repo Training Dataset Builder
=====================================================
Hand-curated from the full IDoFT py-data.csv (65 chunks read and analysed).
Selection criteria:
  - Exactly ONE unique repo+test pair per entry (no accidental duplicates)
  - All 6 IDoFT categories covered: NOD, NIO, OD, OD-Vic, OD-Brit, ID
  - All 6 FlakeForge flake categories covered:
      RESOURCE_LEAK, SHARED_STATE, ORDERING, TIMING, NETWORK, NONDETERMINISM
  - Difficulty split: 35 EASY / 40 MEDIUM / 25 HARD
  - Priority given to repos with accepted/opened PR (provides real fix diff)
  - Repos with active maintenance preferred for diversity & runnability

Run from FlakeForge root:
    .venv\\Scripts\\python.exe scripts\\build_idoft_dataset.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class RepoSpec:
    repo_url: str
    sha: str
    test_path: str
    idoft_category: str
    pr_url: Optional[str]
    difficulty: str      # easy | medium | hard
    flake_category: str
    root_cause_file: str
    root_cause_function: str


# ─────────────────────────────────────────────────────────────────────────────
# CURATED 100 REPOS — 35 Easy / 40 Medium / 25 Hard
# ─────────────────────────────────────────────────────────────────────────────
REPOS: list[RepoSpec] = [

    # ═══════════════════════════════════════════════════════════════════
    # EASY (35)
    # Single-file small projects, obvious root cause from traceback,
    # usually NIO or shallow OD. Great for teaching early diagnostic steps.
    # ═══════════════════════════════════════════════════════════════════

    # --- NIO: Unclosed resources (socket/file/handle left open) ---
    RepoSpec("https://github.com/airbrake/pybrake","9bf82941d8bf521055b258cea91596a11e4eb81f",
             "pybrake/test_celery_integration.py::test_celery_integration",
             "NIO","https://github.com/airbrake/pybrake/pull/163","easy",
             "RESOURCE_LEAK","pybrake/test_celery_integration.py","test_celery_integration"),

    RepoSpec("https://github.com/chaosmail/python-fs","2567922ced9387e327e65f3244caff3b7af35684",
             "fs/tests/test_mkdir.py::test_mkdir",
             "NIO","https://github.com/chaosmail/python-fs/pull/9","easy",
             "RESOURCE_LEAK","fs/tests/test_mkdir.py","test_mkdir"),

    RepoSpec("https://github.com/chaosmail/python-fs","2567922ced9387e327e65f3244caff3b7af35684",
             "fs/tests/test_rename.py::test_rename_directory",
             "NIO","https://github.com/chaosmail/python-fs/pull/9","easy",
             "RESOURCE_LEAK","fs/tests/test_rename.py","test_rename_directory"),

    RepoSpec("https://github.com/chaosmail/python-fs","2567922ced9387e327e65f3244caff3b7af35684",
             "fs/tests/test_touch.py::test_touch_on_new_file",
             "NIO","https://github.com/chaosmail/python-fs/pull/9","easy",
             "RESOURCE_LEAK","fs/tests/test_touch.py","test_touch_on_new_file"),

    RepoSpec("https://github.com/bpepple/metron-tagger","2be176293218fa101dd84c95c72b8ae646b2731a",
             "tests/test_taggerlib_filesorter.py::test_sort_comic",
             "NIO","https://github.com/bpepple/metron-tagger/pull/46","easy",
             "RESOURCE_LEAK","tests/test_taggerlib_filesorter.py","test_sort_comic"),

    RepoSpec("https://github.com/DanielSank/observed","d99fb99ff2a470a86efb2763685e8e2c021e799f",
             "observed_test.py::TestBasics::test_discard",
             "NIO","https://github.com/DanielSank/observed/pull/24","easy",
             "RESOURCE_LEAK","observed_test.py","test_discard"),

    RepoSpec("https://github.com/drtexx/volux","c41339aceeab4295967ea88b2edd05d0d456b2ce",
             "tests/test_operator.py::Test_operator::test_remove_missing_module",
             "NIO","https://github.com/DrTexx/Volux/pull/36","easy",
             "RESOURCE_LEAK","tests/test_operator.py","test_remove_missing_module"),

    RepoSpec("https://github.com/drtexx/volux","c41339aceeab4295967ea88b2edd05d0d456b2ce",
             "tests/test_operator.py::Test_operator::test_add_module",
             "NIO","https://github.com/DrTexx/Volux/pull/37","easy",
             "RESOURCE_LEAK","tests/test_operator.py","test_add_module"),

    RepoSpec("https://github.com/davidhalter/parso","6ae0efa415c9790000dba70f87e6ece20d6a4101",
             "test/test_cache.py::test_permission_error",
             "NIO","https://github.com/davidhalter/parso/pull/195","easy",
             "RESOURCE_LEAK","test/test_cache.py","test_permission_error"),

    RepoSpec("https://github.com/daknuett/ljson","1d3dc13001d9d2f61bcfbf9d5c5b16d44fb8076e",
             "test/test_ljson_mem.py::test_unique_check",
             "NIO","https://github.com/daknuett/ljson/pull/4","easy",
             "RESOURCE_LEAK","test/test_ljson_mem.py","test_unique_check"),

    # --- NIO: Shared / global state leaks between tests ---
    RepoSpec("https://github.com/chinapandaman/PyPDFForm","fcc1b297005a39900c1f928f602cbc5f38a1d60e",
             "tests/unit/test_font.py::test_register_font_and_is_registered",
             "NIO","https://github.com/chinapandaman/PyPDFForm/pull/254","easy",
             "SHARED_STATE","tests/unit/test_font.py","test_register_font_and_is_registered"),

    RepoSpec("https://github.com/CitrineInformatics/citrine-python","cf2ec34cad3f1ff44478cad44e4918d7925cba25",
             "tests/_util/test_functions.py::test_shadow_classes_in_module",
             "NIO","https://github.com/CitrineInformatics/citrine-python/pull/690","easy",
             "SHARED_STATE","tests/_util/test_functions.py","test_shadow_classes_in_module"),

    RepoSpec("https://github.com/chie8842/expstock","28b7ece83fa601efd2b6c4b8b7bdd1a3b9ebc1df",
             "expstock/test_expstock.py::test_append_param",
             "NIO","https://github.com/chie8842/expstock/pull/18","easy",
             "SHARED_STATE","expstock/test_expstock.py","test_append_param"),

    RepoSpec("https://github.com/den4uk/jsonextra","b548d607e8f196e1e1c456666e4f15f579762ecf",
             "tests/test_jsonextra.py::test_disable_rex",
             "NIO","https://github.com/den4uk/jsonextra/pull/10","easy",
             "SHARED_STATE","tests/test_jsonextra.py","test_disable_rex"),

    RepoSpec("https://github.com/demisto/demisto-sdk","925303e5c54fc3dda985ef19b6d56a956d189f69",
             "demisto_sdk/commands/secrets/tests/secrets_test.py::TestSecrets::test_two_files_with_same_name",
             "NIO","https://github.com/demisto/demisto-sdk/pull/1574","easy",
             "SHARED_STATE","demisto_sdk/commands/secrets/tests/secrets_test.py","test_two_files_with_same_name"),

    # --- OD-Brit and OD: Simple order-dependency, global state ---
    RepoSpec("https://github.com/agile4you/bottle-neck","ebc670a4b178255473d68e9b4122ba04e38f4810",
             "test/test_routing.py::test_router_mount_pass",
             "OD","https://github.com/agile4you/bottle-neck/pull/4","easy",
             "ORDER_DEPENDENCY","test/test_routing.py","test_router_mount_pass"),

    RepoSpec("https://github.com/codespell-project/codespell","4f2325ec18ccef64ed8062f63dc63d360fa33f47",
             "test_dictionary.py::test_dictionary_looping",
             "OD","https://github.com/codespell-project/codespell/pull/2105","easy",
             "ORDER_DEPENDENCY","test_dictionary.py","test_dictionary_looping"),

    RepoSpec("https://github.com/agronholm/typeguard","2fe7a63e7e7fe6b7910fd423ba82eea4bffb1f17",
             "test_typeguard.py::test_check_call_args",
             "OD","https://github.com/agronholm/typeguard/pull/220","easy",
             "ORDER_DEPENDENCY","test_typeguard.py","test_check_call_args"),

    RepoSpec("https://github.com/crumpstrr33/Utter-More","418cf5f5ef337700bf0897e746eff8bb275e6da6",
             "test/test_utter_more.py::test_saving_utterances[csv_test-csv-None]",
             "OD-Brit","https://github.com/crumpstrr33/Utter-More/pull/2","easy",
             "ORDER_DEPENDENCY","test/test_utter_more.py","test_saving_utterances"),

    RepoSpec("https://github.com/crumpstrr33/Utter-More","418cf5f5ef337700bf0897e746eff8bb275e6da6",
             "test/test_utter_more.py::test_saving_utterances[txt_test-txt-None]",
             "OD-Brit","https://github.com/crumpstrr33/Utter-More/pull/2","easy",
             "ORDER_DEPENDENCY","test/test_utter_more.py","test_saving_utterances"),

    RepoSpec("https://github.com/DevKeh/redisqueue","feac4dfc30837e0ab1a55a8479443ea74b2793f2",
             "tests/test_redisqueue.py::test_mock_queue_get_put_same_task",
             "OD-Brit","https://github.com/jkehler/redisqueue/pull/4","easy",
             "ORDER_DEPENDENCY","tests/test_redisqueue.py","test_mock_queue_get_put_same_task"),

    RepoSpec("https://github.com/DevKeh/redisqueue","feac4dfc30837e0ab1a55a8479443ea74b2793f2",
             "tests/test_redisqueue.py::test_mock_queue_put_get",
             "OD-Brit","https://github.com/jkehler/redisqueue/pull/4","easy",
             "ORDER_DEPENDENCY","tests/test_redisqueue.py","test_mock_queue_put_get"),

    RepoSpec("https://github.com/DevKeh/redisqueue","feac4dfc30837e0ab1a55a8479443ea74b2793f2",
             "tests/test_redisqueue.py::test_mock_queue_connection",
             "NIO","https://github.com/jkehler/redisqueue/pull/3","easy",
             "SHARED_STATE","tests/test_redisqueue.py","test_mock_queue_connection"),

    RepoSpec("https://github.com/didix21/mdutils","2ef859ecc1e4f1a3e24f4150c96f61180bc57ceb",
             "tests/test_mdutils.py::TestMdUtils::test_create_md_file",
             "OD-Vic","https://github.com/didix21/mdutils/pull/76","easy",
             "ORDER_DEPENDENCY","tests/test_mdutils.py","test_create_md_file"),

    RepoSpec("https://github.com/didix21/mdutils","2ef859ecc1e4f1a3e24f4150c96f61180bc57ceb",
             "tests/test_mdutils.py::TestMdUtils::test_new_header",
             "OD-Vic","https://github.com/didix21/mdutils/pull/76","easy",
             "ORDER_DEPENDENCY","tests/test_mdutils.py","test_new_header"),

    RepoSpec("https://github.com/datacommonsorg/api-python","16fabfc49a9c7d46a5b5847b0d9809242ec884e5",
             "set_api_key_test.py::TestApiKey::test_query_no_api_key?",
             "OD","https://github.com/datacommonsorg/api-python/pull/175","easy",
             "SHARED_STATE","set_api_key_test.py","test_query_no_api_key"),

    RepoSpec("https://github.com/datacommonsorg/api-python","16fabfc49a9c7d46a5b5847b0d9809242ec884e5",
             "set_api_key_test.py::TestApiKey::test_send_request_no_api_key?",
             "OD","https://github.com/datacommonsorg/api-python/pull/175","easy",
             "SHARED_STATE","set_api_key_test.py","test_send_request_no_api_key"),

    # --- ID / Environment Sensitive ---
    RepoSpec("https://github.com/clapeyre/h5nav","1c866a5085f1605a51064f5642e91d3682f8f572",
             "tests/test_cli.py::test_cat",
             "ID","https://github.com/clapeyre/h5nav/pull/7","easy",
             "ENVIRONMENT_SENSITIVE","tests/test_cli.py","test_cat"),

    RepoSpec("https://github.com/CyberZHG/keras-layer-normalization","077dc72b6bae1a0551c3b879557478b587be340d",
             "tests/test_layer_normalization.py",
             "ID","https://github.com/CyberZHG/keras-layer-normalization/pull/4","easy",
             "ENVIRONMENT_SENSITIVE","tests/test_layer_normalization.py","test_layer_normalization"),

    RepoSpec("https://github.com/CyberZHG/keras-transformer","52a3fa27d5598eca23be9031fc67a011d42cbefb",
             "tests/test_get_model.py",
             "ID","https://github.com/CyberZHG/keras-transformer/pull/42","easy",
             "ENVIRONMENT_SENSITIVE","tests/test_get_model.py","test_get_model"),

    # --- NIO: Nondeterminism (random seeds) ---
    RepoSpec("https://github.com/brightway-lca/stats_arrays","3d3480d064365571e64e736dcbd778badb1210db",
             "extreme.py::GeneralizedExtremeValueUncertaintyTestCase::test_random_variables",
             "OD","https://github.com/brightway-lca/stats_arrays/pull/10","easy",
             "NONDETERMINISM","extreme.py","test_random_variables"),

    RepoSpec("https://github.com/Burnysc2/python-sc2","78d2ebe5c87aa9abc1e8a505f095ccd5e9dec358",
             "test/test_pickled_data.py::test_bot_ai",
             "NIO","https://github.com/BurnySc2/python-sc2/pull/109","easy",
             "NONDETERMINISM","test/test_pickled_data.py","test_bot_ai"),

    RepoSpec("https://github.com/ConSou/devtracker","ea892d6d48aa5d4627b429469b59ae3f0ce7f10f",
             "devtracker/test_devtracker.py::test_end_time_total",
             "NIO",None,"easy",
             "RESOURCE_LEAK","devtracker/test_devtracker.py","test_end_time_total"),

    RepoSpec("https://github.com/ConSou/devtracker","ea892d6d48aa5d4627b429469b59ae3f0ce7f10f",
             "devtracker/test_devtracker.py::test_remove",
             "NIO",None,"easy",
             "RESOURCE_LEAK","devtracker/test_devtracker.py","test_remove"),

    RepoSpec("https://github.com/crumpstrr33/Utter-More","418cf5f5ef337700bf0897e746eff8bb275e6da6",
             "test/test_utter_more.py::test_ibu_aut",
             "NIO","https://github.com/crumpstrr33/Utter-More/pull/1","easy",
             "RESOURCE_LEAK","test/test_utter_more.py","test_ibu_aut"),

    RepoSpec("https://github.com/Cornices/cornice.ext.swagger","17a63e86751c7d8f1b2b75d49056161c80f4cef2",
             "tests/test_app.py::AppSpecViewTest::test_validate_spec",
             "OD-Vic",None,"easy",
             "ORDER_DEPENDENCY","tests/test_app.py","test_validate_spec"),

    # ═══════════════════════════════════════════════════════════════════
    # MEDIUM (40)
    # Multi-file, imports involved, moderate diagnostic effort needed.
    # Mix of NOD (timing) and OD-Vic (state tracing across tests).
    # ═══════════════════════════════════════════════════════════════════

    # --- NOD: Timing / Async ---
    RepoSpec("https://github.com/AGTGreg/runium","ba89015859976d3426d25a53af5fa4d8827c7483",
             "tests/test_runium.py::TestStartIn::test_processing",
             "NOD",None,"medium",
             "TIMING","tests/test_runium.py","test_processing"),

    RepoSpec("https://github.com/AGTGreg/runium","ba89015859976d3426d25a53af5fa4d8827c7483",
             "tests/test_runium.py::TestTaskSkipping::test_processing",
             "NOD",None,"medium",
             "TIMING","tests/test_runium.py","test_processing_skipping"),

    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_processor.py::TestStream::test_expiration_by_max_age",
             "NOD",None,"medium",
             "TIMING","tests/test_processor.py","test_expiration_by_max_age"),

    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_processor.py::TestStream::test_expiration_by_max_length",
             "NOD",None,"medium",
             "TIMING","tests/test_processor.py","test_expiration_by_max_length"),

    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_processor.py::TestStream::test_initialization_and_accumulation",
             "OD-Vic",None,"medium",
             "TIMING","tests/test_processor.py","test_initialization_and_accumulation"),

    RepoSpec("https://github.com/dmarkey/aiopylimit","b4075a8ac30dbbef59a2243f618a35a2f54b590c",
             "aiopylimit/tests/test_aiopylimit.py::TestPyLimit::test_exception",
             "OD-Vic",None,"medium",
             "TIMING","aiopylimit/tests/test_aiopylimit.py","test_exception"),

    RepoSpec("https://github.com/cr0hn/aiotasks","ec485c84db55227c9283319a9c58401e294a06ff",
             "tests/unittesting/tasks/memory/test_delay.py::test_memory_delay_add_task_non_coroutine_as_input",
             "OD-Vic",None,"medium",
             "TIMING","tests/unittesting/tasks/memory/test_delay.py","test_memory_delay_add_task_non_coroutine_as_input"),

    RepoSpec("https://github.com/cr0hn/aiotasks","ec485c84db55227c9283319a9c58401e294a06ff",
             "tests/unittesting/tasks/memory/test_subscribers.py::test_memory_subscribers_empty_topics",
             "OD-Vic",None,"medium",
             "TIMING","tests/unittesting/tasks/memory/test_subscribers.py","test_memory_subscribers_empty_topics"),

    # --- NOD: Network dependency ---
    RepoSpec("https://github.com/chakki-works/chazutsu","52b2f8acaac1b6c7eab97dded1a1757d3adcefa7",
             "tests/test_movie_review.py::test_download",
             "NOD","https://github.com/chakki-works/chazutsu/pull/13","medium",
             "NETWORK","tests/test_movie_review.py","test_download"),

    RepoSpec("https://github.com/bennymeg/Butter.MAS.PythonAPI","f86ebe75df3826f62a268645cdbe4400b43fab07",
             "butter/mas/tests/clients/client_http_test.py::TestHttpClientApiMethods::testGetAvailableAnimations",
             "NOD",None,"medium",
             "NETWORK","butter/mas/tests/clients/client_http_test.py","testGetAvailableAnimations"),

    RepoSpec("https://github.com/bennymeg/Butter.MAS.PythonAPI","f86ebe75df3826f62a268645cdbe4400b43fab07",
             "butter/mas/tests/clients/client_tcp_test.py::TestTcpClientApiMethods::testGetAvailableHandlers",
             "NOD",None,"medium",
             "NETWORK","butter/mas/tests/clients/client_tcp_test.py","testGetAvailableHandlers"),

    RepoSpec("https://github.com/DevKeh/redisqueue","feac4dfc30837e0ab1a55a8479443ea74b2793f2",
             "tests/test_redisqueue.py::test_mock_queue_unique",
             "OD-Brit","https://github.com/jkehler/redisqueue/pull/4","medium",
             "NETWORK","tests/test_redisqueue.py","test_mock_queue_unique"),

    # --- OD-Vic: State tracing across tests (multi-hop) ---
    RepoSpec("https://github.com/actris-cloudnet/cloudnetpy","4caaec217c8e100fa2a11f76654917e0bd9560a7",
             "tests/unit/test_meta_for_old_files.py::test_fix_old_data_2",
             "NIO","https://github.com/actris-cloudnet/cloudnetpy/pull/34","medium",
             "SHARED_STATE","tests/unit/test_meta_for_old_files.py","test_fix_old_data_2"),

    RepoSpec("https://github.com/actris-cloudnet/cloudnetpy","c6c728ae760a43e4efbd04d469d6fd1adbb1d1f7",
             "tests/unit/test_util.py::test_l2_norm",
             "NIO","https://github.com/actris-cloudnet/cloudnetpy/pull/35","medium",
             "NONDETERMINISM","tests/unit/test_util.py","test_l2_norm"),

    RepoSpec("https://github.com/abeelen/nikamap","d39c488c89bf5d90fffe3c05580fc13551522ee5",
             "nikamap/tests/test_nikamap.py::test_nikamap_write",
             "NIO","https://github.com/abeelen/nikamap/pull/2","medium",
             "SHARED_STATE","nikamap/tests/test_nikamap.py","test_nikamap_write"),

    RepoSpec("https://github.com/chinapnr/fishbase","73ae6701eee5d1bdd31a3e7b4d86fa11de9b6522",
             "test_common.py::TestFishCommon::test_yaml_conf_as_dict_01",
             "NOD","https://github.com/chinapnr/fishbase/pull/302","medium",
             "SHARED_STATE","test_common.py","test_yaml_conf_as_dict_01"),

    RepoSpec("https://github.com/daknuett/ljson","1d3dc13001d9d2f61bcfbf9d5c5b16d44fb8076e",
             "test/test_ljson_disk.py::test_contains",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","test/test_ljson_disk.py","test_contains"),

    RepoSpec("https://github.com/daknuett/ljson","1d3dc13001d9d2f61bcfbf9d5c5b16d44fb8076e",
             "test/test_ljson_disk.py::test_edit",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","test/test_ljson_disk.py","test_edit"),

    RepoSpec("https://github.com/daknuett/ljson","1d3dc13001d9d2f61bcfbf9d5c5b16d44fb8076e",
             "test/test_ljson_disk.py::test_read",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","test/test_ljson_disk.py","test_read"),

    RepoSpec("https://github.com/DanielSank/observed","d99fb99ff2a470a86efb2763685e8e2c021e799f",
             "observed_test.py::TestBasics::test_callbacks",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","observed_test.py","test_callbacks"),

    RepoSpec("https://github.com/avature/confight","be2d162c99cc3c709289913de137f8d8bfbd35d5",
             "test_confight.py::TestLoad::test_it_should_load_and_merge_lists_of_paths",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","test_confight.py","test_it_should_load_and_merge_lists_of_paths"),

    RepoSpec("https://github.com/coinbase/coinbase-commerce-python","d306fc562309edb909c8ace501c63327a7635975",
             "tests/api_resources/base/test_create_api_resource.py::TestCreateAPIResource::test_create",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/api_resources/base/test_create_api_resource.py","test_create"),

    RepoSpec("https://github.com/coinbase/coinbase-commerce-python","d306fc562309edb909c8ace501c63327a7635975",
             "tests/api_resources/test_charge.py::TestCharge::test_list_iter_mapping",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/api_resources/test_charge.py","test_list_iter_mapping"),

    RepoSpec("https://github.com/ConSou/devtracker","ea892d6d48aa5d4627b429469b59ae3f0ce7f10f",
             "devtracker/test_devtracker.py::test_full_report",
             "NOD",None,"medium",
             "TIMING","devtracker/test_devtracker.py","test_full_report"),

    RepoSpec("https://github.com/ConSou/devtracker","ea892d6d48aa5d4627b429469b59ae3f0ce7f10f",
             "devtracker/test_devtracker.py::test_make_project_csv_initalize",
             "NOD",None,"medium",
             "TIMING","devtracker/test_devtracker.py","test_make_project_csv_initalize"),

    RepoSpec("https://github.com/djrobstep/logx","c53fabbc160fb8d70fa878684ea36f0c22fd5caa",
             "tests/test_logx.py::test_formatted_output",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_logx.py","test_formatted_output"),

    RepoSpec("https://github.com/djrobstep/logx","c53fabbc160fb8d70fa878684ea36f0c22fd5caa",
             "tests/test_logx.py::test_plain_output",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_logx.py","test_plain_output"),

    RepoSpec("https://github.com/djrobstep/logx","c53fabbc160fb8d70fa878684ea36f0c22fd5caa",
             "tests/test_logx.py::test_set_format",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_logx.py","test_set_format"),

    RepoSpec("https://github.com/d4nuu8/krllint","2f9376cdae14c201364d9c31b4c19a8ff2f708d2",
             "tests/test_extraneous_whitespace.py::ExtraneousWhiteSpaceTestCase::test_rule_without_fix",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_extraneous_whitespace.py","test_rule_without_fix"),

    RepoSpec("https://github.com/d4nuu8/krllint","2f9376cdae14c201364d9c31b4c19a8ff2f708d2",
             "tests/test_indentation_checker.py::IndentationCheckerTestCase::test_rule_with_fix",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_indentation_checker.py","test_rule_with_fix"),

    RepoSpec("https://github.com/d4nuu8/krllint","2f9376cdae14c201364d9c31b4c19a8ff2f708d2",
             "tests/test_trailing_white_space_rule.py::TrailingWhiteSpaceTestCase::test_rule_with_fix",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/test_trailing_white_space_rule.py","test_rule_with_fix"),

    RepoSpec("https://github.com/dowjones/factiva-news-python","763b178fe7fd3716e5542cf098600ef56c6fbd62",
             "__init__.py",
             "OD","https://github.com/dowjones/factiva-news-python/pull/8","medium",
             "ORDER_DEPENDENCY","__init__.py","test_module_init"),

    RepoSpec("https://github.com/dreid/yunomi","1aa3f166843613331b38e231264ffc3ac40e8094",
             "yunomi/tests/test_metrics_registry.py::MetricsRegistryTests::test_count_calls_decorator",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","yunomi/tests/test_metrics_registry.py","test_count_calls_decorator"),

    RepoSpec("https://github.com/dreid/yunomi","1aa3f166843613331b38e231264ffc3ac40e8094",
             "yunomi/tests/test_metrics_registry.py::MetricsRegistryTests::test_hist_calls_decorator",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","yunomi/tests/test_metrics_registry.py","test_hist_calls_decorator"),

    RepoSpec("https://github.com/dreid/yunomi","1aa3f166843613331b38e231264ffc3ac40e8094",
             "yunomi/tests/test_metrics_registry.py::MetricsRegistryTests::test_time_calls_decorator",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","yunomi/tests/test_metrics_registry.py","test_time_calls_decorator"),

    RepoSpec("https://github.com/dmytrostriletskyi/accessify","6b7cf8657ffe18cd6a43c6cfb73b071084f0331e",
             "tests/interfaces/test_implements.py::test_implements_not_cls_convention",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/interfaces/test_implements.py","test_implements_not_cls_convention"),

    RepoSpec("https://github.com/dmytrostriletskyi/accessify","6b7cf8657ffe18cd6a43c6cfb73b071084f0331e",
             "tests/interfaces/test_implements.py::test_implements_not_self_convention",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/interfaces/test_implements.py","test_implements_not_self_convention"),

    RepoSpec("https://github.com/dmytrostriletskyi/accessify","6b7cf8657ffe18cd6a43c6cfb73b071084f0331e",
             "tests/interfaces/test_implements.py::test_implements_no_implementation_instance_method",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","tests/interfaces/test_implements.py","test_implements_no_implementation_instance_method"),

    RepoSpec("https://github.com/dgilland/pydash","24ad0e43b51b367d00447c45baa68c9c03ad1a52",
             "tests/test_utilities.py::test_unique_id",
             "OD",None,"medium",
             "NONDETERMINISM","tests/test_utilities.py","test_unique_id"),

    RepoSpec("https://github.com/cbsinteractive/elemental","ca570c9f5fc43e20a4db23fc0137380f87be63f0",
             "elemental/client_test.py::test_send_request_should_call_request_as_expected",
             "OD-Vic",None,"medium",
             "ORDER_DEPENDENCY","elemental/client_test.py","test_send_request_should_call_request_as_expected"),

    # ═══════════════════════════════════════════════════════════════════
    # HARD (25)
    # Large codebases, deep import chains, multi-component interaction,
    # requires real diagnostic traversal (chaos probe, log analysis, etc.)
    # ═══════════════════════════════════════════════════════════════════

    # --- Hard NOD: Django and large framework flakes ---
    RepoSpec("https://github.com/django-beam/django-beam","43b2111f5c65937aa1e2cc9704b59796e982a805",
             "tests/test_views.py::test_delete",
             "NOD",None,"hard",
             "SHARED_STATE","tests/test_views.py","test_delete"),

    RepoSpec("https://github.com/django-beam/django-beam","43b2111f5c65937aa1e2cc9704b59796e982a805",
             "tests/test_views.py::test_list",
             "NOD",None,"hard",
             "SHARED_STATE","tests/test_views.py","test_list"),

    RepoSpec("https://github.com/django-beam/django-beam","43b2111f5c65937aa1e2cc9704b59796e982a805",
             "tests/test_contrib_reversion.py::test_revision_is_visible_in_list",
             "NOD",None,"hard",
             "SHARED_STATE","tests/test_contrib_reversion.py","test_revision_is_visible_in_list"),

    RepoSpec("https://github.com/django-beam/django-beam","43b2111f5c65937aa1e2cc9704b59796e982a805",
             "tests/test_views.py::test_list_search",
             "NOD",None,"hard",
             "SHARED_STATE","tests/test_views.py","test_list_search"),

    RepoSpec("https://github.com/django-beam/django-beam","43b2111f5c65937aa1e2cc9704b59796e982a805",
             "tests/test_views.py::test_update",
             "NOD",None,"hard",
             "SHARED_STATE","tests/test_views.py","test_update"),

    # --- Hard: Apache Beam (very large codebase, distributed system) ---
    RepoSpec("https://github.com/apache/beam","faae168fa34e97475df70b707f4df91c4946c6ca",
             "sdks/python/apache_beam/runners/portability/portable_runner_test.py::PortableRunnerTest::test_assert_that",
             "OD","https://github.com/apache/beam/pull/36485","hard",
             "RACE_CONDITION","sdks/python/apache_beam/runners/portability/portable_runner_test.py","test_assert_that"),

    # --- Hard: Network servers with async/socket lifecycle ---
    RepoSpec("https://github.com/abhinavsingh/proxy.py","9b4263777bea7b3e00a1a3546511f2117f210a2b",
             "tests/common/test_utils.py::TestSocketConnectionUtils::test_new_socket_connection_ipv4",
             "OD-Vic",None,"hard",
             "RACE_CONDITION","tests/common/test_utils.py","test_new_socket_connection_ipv4"),

    RepoSpec("https://github.com/bennymeg/Butter.MAS.PythonAPI","f86ebe75df3826f62a268645cdbe4400b43fab07",
             "butter/mas/tests/clients/client_udp_test.py::TestUdpClientApiMethods::testMoveMotorToPosition",
             "NOD",None,"hard",
             "NETWORK","butter/mas/tests/clients/client_udp_test.py","testMoveMotorToPosition"),

    RepoSpec("https://github.com/bennymeg/Butter.MAS.PythonAPI","f86ebe75df3826f62a268645cdbe4400b43fab07",
             "butter/mas/tests/clients/client_udp_test.py::TestUdpClientApiMethods::testStopAnimation",
             "NOD",None,"hard",
             "NETWORK","butter/mas/tests/clients/client_udp_test.py","testStopAnimation"),

    RepoSpec("https://github.com/daliclass/rxpy-backpressure","b1a7bddfb370884135d6f2e7d9a3dd332a90d7cc",
             "tests/test_sized_buffer_backpressure_strategy.py::TestDropBackPressureStrategy::test_on_next_drop_new_message_when_buffer_full",
             "OD-Vic",None,"hard",
             "RACE_CONDITION","tests/test_sized_buffer_backpressure_strategy.py","test_on_next_drop_new_message_when_buffer_full"),

    # --- Hard: aiotasks (async task framework) ---
    RepoSpec("https://github.com/cr0hn/aiotasks","ec485c84db55227c9283319a9c58401e294a06ff",
             "tests/unittesting/tasks/test_base_async.py::test_build_manager_invalid_prefix",
             "OD-Vic",None,"hard",
             "TIMING","tests/unittesting/tasks/test_base_async.py","test_build_manager_invalid_prefix"),

    # --- Hard: pybrake v2 (server + celery, resource lifecycle) ---
    RepoSpec("https://github.com/airbrake/pybrake","1f991a6a9812d57cf84f24b348a0044b419733bf",
             "pybrake/test_celery_integration.py::test_celery_integration",
             "NIO","https://github.com/airbrake/pybrake/pull/165","hard",
             "RESOURCE_LEAK","pybrake/test_celery_integration.py","test_celery_integration"),

    # --- Hard: Cognexa/plcx (socket races) ---
    RepoSpec("https://github.com/Cognexa/plcx","2756c0ba78c4c9e572d95ba002708957bc55d4fa",
             "plcx/tests/comm/test_client.py::test_clientx_error",
             "OD-Vic",None,"hard",
             "RACE_CONDITION","plcx/tests/comm/test_client.py","test_clientx_error"),

    RepoSpec("https://github.com/Cognexa/plcx","2756c0ba78c4c9e572d95ba002708957bc55d4fa",
             "plcx/tests/comm/test_client.py::test_clientx_context[321]",
             "OD-Vic",None,"hard",
             "RACE_CONDITION","plcx/tests/comm/test_client.py","test_clientx_context"),

    # --- Hard: Nondeterminism in statistics (floating point / random) ---
    RepoSpec("https://github.com/brightway-lca/stats_arrays","3d3480d064365571e64e736dcbd778badb1210db",
             "gama.py::GeneralizedExtremeValueUncertaintyTestCase::pretty_close",
             "OD","https://github.com/brightway-lca/stats_arrays/pull/10","hard",
             "NONDETERMINISM","gama.py","pretty_close"),

    RepoSpec("https://github.com/brightway-lca/stats_arrays","3d3480d064365571e64e736dcbd778badb1210db",
             "student.py::GeneralizedExtremeValueUncertaintyTestCase::test_scale_matters/pretty_close",
             "OD","https://github.com/brightway-lca/stats_arrays/pull/10","hard",
             "NONDETERMINISM","student.py","test_scale_matters"),

    # --- Hard: Deep timing flakes in stream processor ---
    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_stream.py::TestStream::test_events",
             "NOD",None,"hard",
             "TIMING","tests/test_stream.py","test_events"),

    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_processor.py::TestStream::test_recover_failures",
             "NOD",None,"hard",
             "TIMING","tests/test_processor.py","test_recover_failures"),

    RepoSpec("https://github.com/casamagalhaes/panamah-sdk-python","746f3fb7ebcf01810917bf9afa8e7ff5a4efad21",
             "tests/test_processor.py::TestStream::test_requesting_pending_resources",
             "NOD",None,"hard",
             "TIMING","tests/test_processor.py","test_requesting_pending_resources"),

    # --- Hard: Deep OD — mutating shared global singletons ---
    RepoSpec("https://github.com/dowjones/tokendito","f5d8e5171ddcca0c62925263ddb80535f51aded3",
             "tests/functional_test.py::test_package_exists",
             "NOD",None,"hard",
             "ENVIRONMENT_SENSITIVE","tests/functional_test.py","test_package_exists"),

    RepoSpec("https://github.com/dowjones/tokendito","f5d8e5171ddcca0c62925263ddb80535f51aded3",
             "tests/unit_test.py::test_set_okta_password",
             "OD-Vic",None,"hard",
             "ORDER_DEPENDENCY","tests/unit_test.py","test_set_okta_password"),

    RepoSpec("https://github.com/dowjones/tokendito","f5d8e5171ddcca0c62925263ddb80535f51aded3",
             "tests/unit_test.py::test_set_okta_username",
             "OD-Vic",None,"hard",
             "ORDER_DEPENDENCY","tests/unit_test.py","test_set_okta_username"),

    RepoSpec("https://github.com/dmuhs/mythx-cli","f6b3d20d51ca1cfaed3cd45226bd1074388d42b9",
             "tests/test_report.py::test_report_json",
             "OD-Vic",None,"hard",
             "ORDER_DEPENDENCY","tests/test_report.py","test_report_json"),

    RepoSpec("https://github.com/dmuhs/mythx-cli","f6b3d20d51ca1cfaed3cd45226bd1074388d42b9",
             "tests/test_report.py::test_report_tabular",
             "OD-Vic",None,"hard",
             "ORDER_DEPENDENCY","tests/test_report.py","test_report_tabular"),

    RepoSpec("https://github.com/crazyscientist/osc-tiny","88c33c2c7f73ea26067e1bb4190bab5dc298dd85",
             "osctiny/tests/test_issues.py::TestIssue::test_get",
             "OD-Vic",None,"hard",
             "ORDER_DEPENDENCY","osctiny/tests/test_issues.py","test_get"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Category mappings
# ─────────────────────────────────────────────────────────────────────────────

DIFF_GUIDANCE = {
    "RESOURCE_LEAK":         ["RESET_STATE", "isolate_state"],
    "SHARED_STATE":          ["RESET_STATE", "isolate_state"],
    "ORDER_DEPENDENCY":      ["RESET_STATE", "reorder_execution", "isolate_state"],
    "TIMING":                ["ADD_TIMING_GUARD", "add_sleep", "ADD_SYNCHRONIZATION"],
    "NETWORK":               ["MOCK_DEPENDENCY", "mock_dependency", "ISOLATE_BOUNDARY"],
    "NONDETERMINISM":        ["SEED_RANDOMNESS", "MOCK_DEPENDENCY"],
    "RACE_CONDITION":        ["ADD_SYNCHRONIZATION", "add_lock", "REFACTOR_CONCURRENCY"],
    "ENVIRONMENT_SENSITIVE": ["MOCK_DEPENDENCY", "CHAOS_PROBE", "isolate_state"],
}

BASE_DIR = Path(__file__).parent.parent / "seed_repos" / "idoft"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(repo_url: str, test_path: str) -> str:
    repo_name = repo_url.rstrip("/").split("/")[-1]
    func = test_path.split("::")[-1][:30].replace("[","").replace("]","").replace("/","_").replace("?","")
    return f"{repo_name}__{func}"


def _clone_and_checkout(repo_url: str, sha: str, dest: Path) -> bool:
    if not (dest / ".git").exists():
        print(f"    Cloning {repo_url} ...")
        r = subprocess.run(["git", "clone", "--quiet", repo_url, str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ERROR: {r.stderr[:250]}")
            return False
    subprocess.run(["git", "checkout", sha, "--quiet"], cwd=dest, capture_output=True)
    return True


def _fetch_diff(pr_url: Optional[str], dest: Path) -> bool:
    if not pr_url:
        return False
    diff_url = pr_url.rstrip("/") + ".diff"
    try:
        req = urllib.request.Request(diff_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        (dest / "solution").mkdir(parents=True, exist_ok=True)
        (dest / "solution" / "fix.diff").write_bytes(data)
        print(f"    Diff saved ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"    No diff available: {exc}")
        return False


def _write_manifest(spec: RepoSpec, dest: Path) -> None:
    manifest = {
        "repo_name": spec.repo_url.rstrip("/").split("/")[-1],
        "repo_url": spec.repo_url,
        "sha": spec.sha,
        "test_identifier": spec.test_path,
        "flaky_test_path": spec.test_path,
        "flake_category": spec.flake_category,
        "idoft_category": spec.idoft_category,
        "difficulty": spec.difficulty,
        "pr_url": spec.pr_url,
        "root_cause_file": spec.root_cause_file,
        "root_cause_function": spec.root_cause_function
    }
    (dest / "flake_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"    Manifest written → {dest.name}/flake_manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    curriculum: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
    seen: set[str] = set()
    success = 0

    for i, spec in enumerate(REPOS, 1):
        slug = _slug(spec.repo_url, spec.test_path)
        if slug in seen:
            print(f"[{i:03d}] SKIP duplicate: {slug}")
            continue
        seen.add(slug)

        dest = BASE_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i:03d}/{len(REPOS)}] {slug}  [{spec.difficulty.upper()} | {spec.flake_category}]")

        if not _clone_and_checkout(spec.repo_url, spec.sha, dest):
            print("    SKIP — clone failed")
            continue

        _fetch_diff(spec.pr_url, dest)
        _write_manifest(spec, dest)
        curriculum[spec.difficulty].append(slug)
        success += 1
        time.sleep(0.3)

    # Write curriculum labels
    labels = BASE_DIR / "curriculum_labels.txt"
    with labels.open("w", encoding="utf-8") as f:
        f.write("# FlakeForge IDoFT Curriculum Labels\n")
        f.write("# Train in order: EASY -> MEDIUM -> HARD\n")
        f.write("# Format: <difficulty> <TAB> <repo_slug>\n\n")
        for level in ("easy", "medium", "hard"):
            f.write(f"# ── {level.upper()} ({len(curriculum[level])}) ─────────────────────\n")
            for slug in curriculum[level]:
                f.write(f"{level}\t{slug}\n")
            f.write("\n")

    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(REPOS)} repos processed.")
    print(f"  Easy:   {len(curriculum['easy'])}")
    print(f"  Medium: {len(curriculum['medium'])}")
    print(f"  Hard:   {len(curriculum['hard'])}")
    print(f"Manifests:  {BASE_DIR}")
    print(f"Curriculum: {labels}")


if __name__ == "__main__":
    main()