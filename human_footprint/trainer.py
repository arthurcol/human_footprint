import argparse
import os

import tensorflow as tf

from human_footprint.imagery_dataset import (
    ImageryDatasetParser,
    get_mixer_json,
    list_blobs,
)
from human_footprint.losses import DiceLoss, JaccardLoss
from human_footprint.model import UnetLandcover

PATH_DATASET = "gs://human-footprint/france_2015/"
BUCKET = "human-footprint"
BLOB_JSON = "france_2015/france_2015mixer.json"
BANDS = [f"SR_B{i}" for i in range(2, 5)]
LABEL = "settlement"
PATH_CSV_LOG = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "logs"
)
PATH_CHECKPOINTS = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "checkpoints"
)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experience-name",
        metavar="-x",
        dest="exp_name",
        default="no_name",
    )
    parser.add_argument("--epochs", metavar="-e", dest="epochs", default=10)
    parser.add_argument("--batches", metavar="-b", dest="batch_size", default=16)
    parser.add_argument("--loss", metavar="-l", dest="loss", default="dice")
    parser.add_argument("--val-split", metavar="-vs", dest="val_split", default=0.8)

    return parser


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    # fetch data
    print("Creating tensorflow dataset instance...")
    tf_records = list_blobs("human-footprint", "france_2015/", delimiter="/")
    dataset = tf.data.TFRecordDataset(tf_records, compression_type="GZIP")
    mixer = get_mixer_json(BUCKET, BLOB_JSON)

    parser_fn = ImageryDatasetParser(mixer_file=mixer, band_names=BANDS, label=LABEL)

    parsed_dataset = dataset.map(parser_fn)
    train_dataset = parsed_dataset.take(
        int(parser_fn.patches * float(args.val_split))
    ).batch(int(args.batch_size))
    val_dataset = parsed_dataset.skip(
        int(parser_fn.patches * float(args.val_split))
    ).batch(int(args.batch_size))

    # instantiate model
    print("Instantiate model...")

    model = UnetLandcover(
        encoder="vgg16",
        pretrained=True,
        residual_connexion="concatenate",
        input_shape=(256, 256, 3),
    )

    # loss, optimizer
    if args.loss == "dice":
        loss = DiceLoss()
    elif args.loss == "jaccard":
        loss = JaccardLoss()
    else:
        raise ValueError(
            f"loss must be one of `jaccard` or `dice`. Received {args.loss}"
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # metrics
    iou_score = tf.keras.metrics.BinaryIoU(target_class_ids=[1], name="iou_score")
    recall = tf.keras.metrics.Recall(name="recall")
    precision = tf.keras.metrics.Precision(name="precision")

    # callbacks
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001, verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(PATH_CSV_LOG, f"{args.exp_name}.csv")
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            PATH_CHECKPOINTS,
            "parser.exp_name",
            "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        ),
        monitor="val_iou_score",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    # compile model

    print("Compile model...")
    model.compile(
        optimizer=optimizer, loss=loss, metrics=[iou_score, precision, recall]
    )

    # launch training
    print(f"Training model for {args.epochs} of {args.batch_size} batch size")

    model.fit(
        train_dataset,
        epochs=int(args.epochs),
        callbacks=[lr_scheduler, csv_logger, es, checkpoints],
        validation_data=val_dataset,
        verbose=2,
    )
