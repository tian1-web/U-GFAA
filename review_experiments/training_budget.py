"""Explicit user-selected epoch budget; the original 49-epoch mode stays available."""


def validate_epochs(epochs):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 10:
        raise ValueError('Training budget must be an integer of at least 10 epochs')
    return epochs


def source_epoch(epoch, total_epochs=49):
    validate_epochs(total_epochs)
    return epoch if total_epochs == 49 else epoch * 49.0 / total_epochs


def budget_record(total_epochs):
    validate_epochs(total_epochs)
    return {'total_epochs': total_epochs, 'source_total_epochs': 49,
            'curriculum_epoch_mapping': 'epoch' if total_epochs == 49 else f'epoch * 49 / {total_epochs}',
            'optimizer_stage_2_starts_at_epoch': 10,
            'note': 'User-selected training budget. Only curriculum progress is scaled; optimizer stage boundary and other source settings are retained.'}


def budget_note(total_epochs):
    return (f'User-selected epochs 1 through {total_epochs}; original source budget is 49. '
            'Curriculum progress is scaled to end at source epoch 49; optimizer stage 2 still begins at epoch 10.')


def configured_epochs(config):
    epochs = config['epochs']
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError('Run must define contiguous epochs starting at 1')
    result = validate_epochs(len(epochs))
    if config['config']['train']['n_epochs'] != result + 1:
        raise ValueError('Source loop upper bound and declared budget differ')
    return result
