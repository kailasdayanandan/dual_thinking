
def get_dataset_attributes():
    return ['Figure Ground', 'Proximity', 'Similarity', 'Continuity', 'Amodal', \
                            'Global Mix', 'Size Diff', 'Count Diff', 'Camouflage']

def get_model_name_list():
    return [ 'detectors_htc_r50', 'detectors_htc_r101', 
                    'mask_rcnn_r50', 'mask_rcnn_r101', 'mask_rcnn_x101',
                    'yolact_r50', 'yolact_r101',
                    'swin-t',  'swin-s',                     
                    'instaboost_r50', 'instaboost_r101', 'instaboost_x101', 
                    'groie_r50', 'groie_r101',
                    'gpt-4o', 'gpt-4o-mini', 'llama-3.2-11B'#, 'llama-3.2-90B'
                    ]


def get_model_display_details_by_name(mdl_name_list):

    model_details_all = [
            ('retinanet_r50', 'RetinaNet (R50)', 'RetinaNet-R50'),
            ('retinanet_r101', 'RetinaNet (R101)', 'RetinaNet-R101'),
            ('retinanet_x101', 'RetinaNet (X101)', 'RetinaNet-X101'),
            ('carafe_r50', 'RetinaNet (R50)', 'Carafe-R50'),
            ('gestalt_r50', 'GSTLT (R50)', 'Gestalt-R50'),
            ('rfp_htc_r50', 'RFC (R50)', 'Rec Feat Pyr-R50'),
            ('sac_htc_r50', 'SAC (R50)', 'Swi Atr Conv -R50'),            
            ('detectors_htc_r50', 'D_RS (R50)', 'DetectoRS-R50'),
            ('detectors_htc_r101', 'D_RS (R101)', 'DetectoRS-R101'),
            ('mask_rcnn_r50', 'M-RCNN (R50)', 'M-RCNN-R50'),
            ('mask_rcnn_r101', 'M-RCNN (R101)', 'M-RCNN-R101'),
            ('mask_rcnn_x101', 'M-RCNN (X101)', 'M-RCNN-X101'),
            ('instaboost_r50', 'IBoost (R50)', 'Instaboost-R50'),
            ('instaboost_r101', 'IBoost (R101)', 'Instaboost-R101'),
            ('instaboost_x101', 'IBoost (X101)', 'Instaboost-X101'),
            ('yolact_r50', 'YOLACT (R50)', 'YOLACT-R50'),
            ('yolact_r101', 'YOLACT (R101)', 'YOLACT-R101'),
            ('groie_r50', 'GROIE (R50)', 'GROIE-R50'),
            ('groie_r101', 'GROIE (R101)', 'GROIE-R101'),
            ('simple-copy-paste', 'SCP (R101)', 'Simple Copy Paste'),
            ('swin-t', 'Swin-T', 'Swin-T'),
            ('swin-s', 'Swin-S', 'Swin-S'),            
            ('mask2former_r50', 'M2Former (R50)', 'M2Former-R50'),
            ('mask2former_r101', 'M2Former (R101)', 'M2Former-R101'),
            ('mask2former_swin_t', 'M2Former (ST)', 'M2Former-Swin-T'),
            ('mask2former_swin_s', 'M2Former (SS)', 'M2Former-Swin-S'),
            ('query_inst_r50', 'QueryInst (R50)', 'QueryInst-R50'),
            ('query_inst_r101', 'QueryInst (R101)', 'QueryInst-R101'),
            ('self_sup_r50', 'SelfSup (R50)', 'Self Supervision-R50'),
            ('scnet_r50', 'SC-Net (R50)', 'SC-Net-R50'),
            ('scnet_r101', 'SC-Net (R101)', 'SC-Net-R101'),
            ('scnet_x101', 'SC-Net (X101)', 'SC-Net-X101'),
            ('solov2_r50', 'SOLOv2 (R50)', 'SOLOv2-R50'),
            ('solov2_r101', 'SOLOv2 (R101)', 'SOLOv2-R101'),
            ('solov2_x101', 'SOLOv2 (X101)', 'SOLOv2-X101'), 
            ('gpt-4o', 'GPT-4o', 'GPT-4o'), 
            ('gpt-4o-mini', 'GPT-4o-mini', 'GPT-4o-mini'), 
            ('llama-3.2-11B', 'Llama-3.2-11B', 'Llama-3.2-11B'),   
            ('llama-3.2-90B', 'Llama-3.2-90B', 'Llama-3.2-90B'),                       
        ]

    model_names_master = [x[0] for x in model_details_all]
    model_indexs = [model_names_master.index(x) for x in mdl_name_list]
    model_names_all = [model_details_all[index][0] for index in model_indexs]
    model_display_names_all = [model_details_all[index][1] for index in model_indexs]
    model_display_table_all = [model_details_all[index][2] for index in model_indexs]

    model_display_names_d_all = dict(zip(model_names_all, model_display_names_all))
    model_display_table_d_all = dict(zip(model_names_all, model_display_table_all))

    return [model_details_all[index][0] for index in model_indexs], model_display_names_d_all, model_display_table_d_all



def get_mmdetection_preloaded_details():

    model_cfg = {
        'gestalt_r50' : (
            './mmdetection/configs/gestalt/gestalt_htc_r50_1x_coco.py',            
            'gestalt_htc_r50_1x_coco.pth',
            ''
        ),  
        'mask_rcnn_r50' : (
            './mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py',              
            'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
        ),
        'mask_rcnn_r101' : (
            './mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py',
            'mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth'
        ),
        'mask_rcnn_x101' : (
            './mmdetection/configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco.py',            
            'mask_rcnn_x101_32x8d_fpn_1x_coco_20220630_173841-0aaf329e.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco/mask_rcnn_x101_32x8d_fpn_1x_coco_20220630_173841-0aaf329e.pth'
        ),
        'rfp_htc_r50' : (
            './mmdetection/configs/detectors/htc_r50_rfp_1x_coco.py',
            'htc_r50_rfp_1x_coco-8ff87c51.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco-8ff87c51.pth'
        ),
        'sac_htc_r50' : (
            './mmdetection/configs/detectors/htc_r50_sac_1x_coco.py',
            'htc_r50_sac_1x_coco-bfa60c54.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco-bfa60c54.pth'
        ),
        'detectors_htc_r50' : (
            './mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py',
            'detectors_htc_r50_1x_coco-329b1453.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth'
        ),
        'detectors_htc_r101' : (
            './mmdetection/configs/detectors/detectors_htc_r101_20e_coco.py',                        
            'detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'
        ),
        'yolact_r50' : (
            './mmdetection/configs/yolact/yolact_r50_1x8_coco.py',
            'yolact_r50_1x8_coco_20200908-f38d58df.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth'
        ),
        'yolact_r101' : (
            './mmdetection/configs/yolact/yolact_r101_1x8_coco.py',            
            'yolact_r101_1x8_coco_20200908-4cbe9101.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth'
        ),
        'swin-t' : (
            './mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
            'mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'
        ),
        'swin-s' : (
            './mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
            'mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
        ),
        'detr_r50' : (
            './mmdetection/configs/detr/detr_r50_8x2_150e_coco.py',            
            'detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        ),
        'ssj_scp' : (
            './mmdetection/configs/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco.py',
            'mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229-80ee90b7.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229-80ee90b7.pth'
        ),
        'instaboost_r50' : (
            './mmdetection/configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py',            
            'mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth'
        ),
        'instaboost_r101' : (
            './mmdetection/configs/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco.py',            
            'mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth'
        ),
        'instaboost_x101' : (
            './mmdetection/configs/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco.py',            
            'mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth'
        ),
        'groie_r50' : (
            './mmdetection/configs/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py',            
            'mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth'
        ),
        'groie_r101' : (
            './mmdetection/configs/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py',            
            'mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth'
        ),      
        'mask2former_r50' : (
            './mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py',            
            'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'
        ),        
        'mask2former_r101' : (
            './mmdetection/configs/mask2former/mask2former_r101_lsj_8x2_50e_coco.py',            
            'mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r101_lsj_8x2_50e_coco/mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth'
        ),
        'mask2former_swin_t' : (
            './mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py',            
            'mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037.pth'
        ),           
        'mask2former_swin_s' : (
            './mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py',            
            'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
        ),
        'query_inst_r50' : (
            './mmdetection/configs/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py',            
            'queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_101802-85cffbd8.pth'
        ),
        'query_inst_r101' : (
            './mmdetection/configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py',            
            'queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'
        ),
        'self_sup_r50' : (
            './mmdetection/configs/selfsup_pretrain/mask_rcnn_r50_fpn_swav-pretrain_ms-2x_coco.py',            
            'mask_rcnn_r50_fpn_swav-pretrain_ms-2x_coco_20210605_163717-08e26fca.pth',            
            'https://download.openmmlab.com/mmdetection/v2.0/selfsup_pretrain/mask_rcnn_r50_fpn_swav-pretrain_ms-2x_coco/mask_rcnn_r50_fpn_swav-pretrain_ms-2x_coco_20210605_163717-08e26fca.pth'
        ),   
        'scnet_r50' : (
            './mmdetection/configs/scnet/scnet_r50_fpn_20e_coco.py',            
            'scnet_r50_fpn_20e_coco-a569f645.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco-a569f645.pth'
        ),
        'scnet_r101' : (
            './mmdetection/configs/scnet/scnet_r101_fpn_20e_coco.py',            
            'scnet_r101_fpn_20e_coco-294e312c.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco-294e312c.pth'
        ),
        'scnet_x101' : (
            './mmdetection/configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py',            
            'scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth'
        ),     
        'solov2_r50' : (
            './mmdetection/configs/solov2/solov2_r50_fpn_3x_coco.py',            
            'solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'
        ),
        'solov2_r101' : (
            './mmdetection/configs/solov2/solov2_r101_dcn_fpn_3x_coco.py',            
            'solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth'
        ),
        'solov2_x101' : (
            './mmdetection/configs/solov2/solov2_x101_dcn_fpn_3x_coco.py',            
            'solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth'
        ),
        'retinanet_r50' : (
            './mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py',            
            'retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
        ),
        'retinanet_r101' : (
            './mmdetection/configs/retinanet/retinanet_r101_fpn_2x_coco.py',            
            'retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
        ),
        'retinanet_x101' : (
            './mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py',            
            'retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'
        ),
        'carafe_r50' : (
            './mmdetection/configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py',            
            'faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth',
            'https://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth'
        ),          
    }

    return model_cfg


def get_mmdetection_model_config(model_name):
    model_cfg =  get_mmdetection_preloaded_details()
    if model_name in model_cfg:
        mm_config, mm_weights, mm_url = model_cfg[model_name]
        return mm_config, mm_weights, mm_url
    return None, None, None
