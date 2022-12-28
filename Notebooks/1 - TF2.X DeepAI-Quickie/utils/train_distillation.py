# Copyright 2022 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import os
import numpy as np


class DistillationClassificationTrainer():
    """
    Training class for Distillation on a Classification task. It takes two models: a student and a teacher. It provides methods to train the student for the target classification problem distilling knowledge from the teacher.

    ...

    Attributes
    ----------
    student_model: obj
        student network object
    teacher_model: obj
        teacher network object
    soft_distillation: bool 
        if True, soft distillation is applied instead of hard distillation
    distil_temperature: float
        soft distillation temperature tau
    lambda_perc: float
        soft distillation balancing parameter
    n_classes: int
        number of classes for the specific classification task
    name_model: str
        name of the student model for saving it
    save_model: bool
        save best model
    log_training: bool
        save training logs    
    verbose: bool
        show some info

    Methods
    -------
    compile(self, loss, metric,  optimizer, label_smoothing=0.1, log_dir='logs', bin_dir='bin')
        Compile model, adding loss, metrics, optimizer and paths
    fit(self, ds_train, batch_size=None, epochs=20, evaluate_every="epoch",
            validation_data=None, initial_epoch=0, save_best_only=True, track="accuracy")
        Train a model with the given data
    """

    def __init__(self, student_model, teacher_model, distil_temperature, lambda_perc, n_classes=10,  name_student_model="student_model.h5", name_teacher_model="teacher_model.h5", soft_distillation=True, save_model=False, log_training=False, verbose=True, **kwargs):
        """
        Parameters
        ----------      
        student_model: obj
            student network object
        teacher_model: obj
            teacher network obj
        soft_distillation: bool 
            if True, soft distillation is applied instead of hard distillation
        distil_temperature: float
            soft distillation temperature tau
        lambda_perc: float
            soft distillation balancing parameter
        n_classes: int
            number of classes for the specific classification task
        name_student_model: str
            name of the student model for saving it
        name_teacher_model: str
            name of the teacher model
        save_model: bool
            save best model
        log_training: bool
            save training logs    
        verbose: bool
            show some info
        """
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.s_distil = soft_distillation
        self.T = distil_temperature
        self.LAMBDA = lambda_perc
        self.name_student_model = name_student_model
        self.name_teacher_model = name_teacher_model
        self.n_classes = n_classes
        self.save_model = save_model
        self.log_training = log_training
        self.verbose = verbose

    def compile(self, loss, metric,  optimizer, label_smoothing=0.1, log_dir='logs', bin_dir='bin'):
        """
        Compile model, adding loss, metrics, optimizer and paths

        Parameters
        ----------
        loss: obj
            main loss for the training. Usuall cross-entropy
        metric: obj
            metric to evaluate the network
        label_smoothing: float
            label smoothing for hard distillation
        log_dir: str
            path for training logs
        bin_dir: str
            path for training weights
        """
        self.loss = loss
        self.s_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1)  # for hard distillation
        self.k_loss = tf.keras.losses.KLDivergence()  # for soft disttillation
        self.metric = metric
        self.from_logits = loss.get_config()['from_logits']
        self.bin_dir = os.path.join(bin_dir, self.name_student_model)
        self.bin_teacher_dir = os.path.join(bin_dir, self.name_teacher_model)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(bin_dir):
            os.mkdir(bin_dir)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), loss=tf.Variable(np.inf),
                                              accuracy=tf.Variable(0.), optimizer=optimizer, model=self.student_model)

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, max_to_keep=1, directory=self.bin_dir)

    def fit(self, ds_train, batch_size=None, epochs=20, evaluate_every="epoch",
            validation_data=None, initial_epoch=0, save_best_only=True, track="accuracy"):
        """
        Train a model with the given data

        Parameters
        ----------
        ds_train: obj
            training data
        batch_size: int
            batch training dimension
        epochs: int
            number of epochs
        evaluate_every: str
            evaluate with validation every "step" or "epoch"
        validation_data: obj
            validation data
        initial_epoch: int
            starting epoch if a training is resumed
        save_best_only: bool
            if save always or only best weights (student weights)
        track: str
            track accuracy or loss for saving weights
        """
        track = track.lower()
        if track not in ["accuracy", "loss"]:
            raise ValueError(f"Cannot track {track}.")

        train_ds = ds_train  # I don't want to swap. lol
        if self.log_training:
            writer_train = tf.summary.create_file_writer(
                os.path.join(self.log_dir, f'train'))

        stateful_metrics = ['Loss', 'Accuracy']
        loss_to_track = tf.constant(np.inf)
        accuracy_to_track = tf.constant(0.)

        if validation_data is not None:
            val_ds = validation_data  # here too
            if self.log_training:
                writer_val = tf.summary.create_file_writer(
                    os.path.join(self.log_dir, f'val'))
            stateful_metrics.append('Val Loss')
            stateful_metrics.append('Val Accuracy')

        total_step = tf.cast(self.checkpoint.step, tf.int64)
        steps_per_epoch = len(ds_train)

        if validation_data is not None:
            if evaluate_every == "step":
                evaluate_every = 1
            elif evaluate_every == "epoch":
                evaluate_every = steps_per_epoch
            else:
                if not isinstance(evaluate_every, int):
                    raise ValueError(
                        f'Wrong "evaluate_every": {evaluate_every}. Acceptable values are "step", "epoch" or int.')
                else:
                    evaluate_every = min(evaluate_every, steps_per_epoch)
        if self.verbose:
            print(
                f"Validating validation dataset every {evaluate_every} steps.")

        for epoch in range(epochs - initial_epoch):
            print("\nEpoch {}/{}".format(epoch + 1 + initial_epoch, epochs))
            pb_i = tf.keras.utils.Progbar(
                steps_per_epoch, stateful_metrics=stateful_metrics)

            for step, (x_train_step, y_train_step) in enumerate(train_ds):
                if step == 0:  # a new epoch is starting -> reset metrics
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()

                total_step += 1

                self._train_step(x_train_step, y_train_step)

                self.checkpoint.step.assign_add(1)

                if self.log_training:
                    with writer_train.as_default():
                        tf.summary.scalar(
                            'Accuracy', self.train_accuracy.result(), step=total_step)
                        tf.summary.scalar(
                            'Loss', self.train_loss.result(), step=total_step)
                    writer_train.flush()

                values = [('Loss', self.train_loss.result()),
                          ('Accuracy', self.train_accuracy.result())]

                if validation_data is not None:
                    if step != 0 and ((step + 1) % evaluate_every) == 0:
                        self.val_loss.reset_states()
                        self.val_accuracy.reset_states()

                        for x_val_step, y_val_step in val_ds:
                            self._test_step(x_val_step, y_val_step)

                        if self.log_training:
                            with writer_val.as_default():
                                tf.summary.scalar(
                                    'Accuracy', self.val_accuracy.result(), step=total_step)
                                tf.summary.scalar(
                                    'Loss', self.val_loss.result(), step=total_step)
                            writer_val.flush()

                        values.append(('Val Loss', self.val_loss.result()))
                        values.append(
                            ('Val Accuracy', self.val_accuracy.result()))

                        loss_to_track = self.val_loss.result()
                        accuracy_to_track = self.val_accuracy.result()
                else:  # if validation is not available, track training
                    loss_to_track = self.train_loss.result()
                    accuracy_to_track = self.train_accuracy.result()

                pb_i.add(1, values=values)  # update bar

                if save_best_only:
                    if (track == "loss" and loss_to_track >= self.checkpoint.loss) or \
                       (track == "accuracy" and accuracy_to_track <= self.checkpoint.accuracy):  # no improvement, skip saving checkpoint
                        continue

                if self.save_model:
                    self.checkpoint.loss = loss_to_track
                    self.checkpoint.accuracy = accuracy_to_track
                    self.student_model.save_weights(self.bin_dir)
                    sellf.teacher_model.save_weights(self.bin_teacher_dir)

    @tf.function
    def _train_step(self, x, y_true):
        """
        Private function to execute a training step with RSC regularizer.
        """
        if self.s_distil:
            # teacher predictions with softmax and a certain temperature
            y_pred_logits_t = tf.nn.softmax(
                self.teacher_model(x)/self.T, axis=-1)

            with tf.GradientTape() as tape:
                y_pred_logits = self.student_model(x)
                y_pred_class = tf.nn.softmax(y_pred_logits, axis=-1)
                y_pred_distil = tf.nn.softmax(y_pred_logits/self.T, axis=-1)

                if self.from_logits:
                    loss = (1 - self.LAMBDA)*self.loss(y_true, y_pred_logits) + \
                        (self.T**2)*self.LAMBDA * \
                        self.k_loss(y_pred_logits_t, y_pred_distil)
                else:
                    loss = (1 - self.LAMBDA)*self.loss(y_true, y_pred_class) + \
                        (self.T**2)*self.LAMBDA * \
                        self.k_loss(y_pred_logits_t, y_pred_distil)

            # compute gradients respect to model variables
            gradients = tape.gradient(
                loss, self.checkpoint.model.trainable_variables)

            self.checkpoint.optimizer.apply_gradients(
                zip(gradients, self.checkpoint.model.trainable_variables))

        else:

            y_pred_t = tf.argmax(tf.nn.softmax(
                self.teacher_model(x), axis=-1), axis=-1)
            y_pred_t = tf.one_hot(y_pred_t, self.n_classes)
            with tf.GradientTape() as tape:
                y_pred_logits = self.student_model(x)
                y_pred_class = tf.nn.softmax(y_pred_logits, axis=-1)
                y_pred_distil = tf.nn.softmax(y_pred_logits, axis=-1)

                if self.from_logits:
                    loss = 0.5*self.loss(y_true, y_pred_logits) + \
                        0.5*self.s_loss(y_pred_t, y_pred_distil)
                else:
                    loss = 0.5*self.loss(y_true, y_pred_class) + \
                        0.5*self.s_loss(y_pred_t, y_pred_distil)

            # compute gradients respect to model variables
            gradients = tape.gradient(
                loss, self.checkpoint.model.trainable_variables)

            self.checkpoint.optimizer.apply_gradients(
                zip(gradients, self.checkpoint.model.trainable_variables))

        self.metric.reset_states()
        metric = self.metric(y_true, y_pred_class)
        self.train_loss(loss)
        self.train_accuracy(metric)

    @tf.function
    def _test_step(self, x, y_true):
        """
        Private function to execute a testing step with the model during the training phase.
        """
        y_pred_logits = self.student_model(x)
        y_pred = tf.nn.softmax(y_pred_logits, axis=-1)

        loss = self.loss(y_true, y_pred)
        self.metric.reset_states()
        metric = self.metric(y_true, y_pred)

        self.val_loss(loss)
        self.val_accuracy(metric)
