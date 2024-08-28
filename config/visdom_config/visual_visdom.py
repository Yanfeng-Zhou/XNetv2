from visdom import Visdom
import os

def visdom_initialization_XNetv2(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup1', 'Train Sup2', 'Train Sup3', 'Train Unsup'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc1'], width=550, height=350))
    visdom.line([[0., 0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup1', 'Val Sup2', 'Val Sup3'], width=550, height=350))
    visdom.line([0.], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc1'], width=550, height=350))
    return visdom

def visualization_XNetv2(vis, epoch, train_loss, train_loss_sup1, train_loss_sup2, train_loss_sup3, train_loss_unsup, train_m_jc1, val_loss_sup1, val_loss_sup2, val_loss_sup3, val_m_jc1):
    vis.line([[train_loss, train_loss_sup1, train_loss_sup2, train_loss_sup3, train_loss_unsup]], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc1], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_sup1, val_loss_sup2, val_loss_sup3]], [epoch], win='val_loss', update='append')
    vis.line([val_m_jc1], [epoch], win='val_jc', update='append')

def visual_image_sup(vis, mask_train, pred_train, mask_val, pred_val):

    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis'))
    vis.heatmap(pred_train, win='train_pred1', opts=dict(title='Train Pred', colormap='Viridis'))
    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis'))
    vis.heatmap(pred_val, win='val_pred1', opts=dict(title='Val Pred', colormap='Viridis'))