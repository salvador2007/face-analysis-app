# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import cv2
import warnings
import logging
import shutil
import hashlib
import gc

warnings.filterwarnings('ignore')

# ========== 로깅 개선 ==============
def setup_logging():
    """로깅 설정 개선"""
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 새로운 로깅 설정
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========== GPU 최적화 클래스 개선 ==============
class GPUOptimizer:
    @staticmethod
    def setup_gpu_optimized(memory_limit: int = 6144) -> bool:  # 메모리 제한 감소
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu = gpus[0]
                # 메모리 증가 허용
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # 메모리 제한 설정
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                
                # Mixed Precision 설정 (메모리 효율성 증대)
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                logger.info(f"✅ GPU 최적화 완료: {memory_limit}MB 제한")
                logger.info(f"✅ Mixed Precision 활성화: {policy.name}")
                
                # GPU 테스트
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor).numpy()
                    logger.info(f"✅ GPU 테스트 성공: {result}")
                return True
            else:
                logger.info("ℹ️ GPU를 찾을 수 없습니다. CPU로 학습합니다.")
                return False
        except Exception as e:
            logger.error(f"⚠️ GPU 설정 중 오류: {e}")
            logger.info("CPU 모드로 전환합니다.")
            return False

    @staticmethod
    def check_tensorflow_gpu() -> bool:
        logger.info("\n🔍 TensorFlow GPU 설정 확인:")
        logger.info("=" * 40)
        logger.info(f"TensorFlow 버전: {tf.__version__}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU 장치: {len(gpu_devices)}개")
        
        if gpu_devices:
            for i, gpu in enumerate(gpu_devices):
                logger.info(f"  GPU {i}: {gpu}")
                
        logger.info(f"CUDA 지원: {tf.test.is_built_with_cuda()}")
        logger.info(f"GPU 사용 가능: {len(gpu_devices) > 0}")
        return len(gpu_devices) > 0

# ========== 모델 빌더 클래스 개선 ==============
class ModelBuilder:
    @staticmethod
    def create_efficient_model(input_shape=(224,224,3), num_classes=7, dropout_rate=0.3):
        """더 안정적이고 효율적인 모델 구조"""
        inputs = keras.Input(shape=input_shape, name='face_input')
        
        # 데이터 정규화를 별도 레이어로 분리
        x = layers.Rescaling(1./255)(inputs)
        
        # 데이터 증강 (훈련시에만 적용)
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # 블록1 - 가벼운 시작
        x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # 블록2
        x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # 블록3
        x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # 블록4 - 더 효율적인 구조
        x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense 레이어 간소화
        x = layers.Dense(256, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate * 1.5)(x)
        
        x = layers.Dense(128, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer - Mixed Precision 호환성 확보
        outputs = layers.Dense(
            num_classes, 
            activation='softmax', 
            kernel_initializer='glorot_uniform', 
            dtype='float32',  # 명시적 float32 설정
            name='predictions'
        )(x)
        
        model = keras.Model(inputs, outputs, name='EfficientFaceClassifier')
        return model

# ========== 데이터 관리 클래스 개선 ==============
class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    def validate_data_structure(self):
        """데이터 구조 검증 개선"""
        if not os.path.exists(self.data_dir):
            logger.error(f"❌ 데이터 디렉터리가 존재하지 않습니다: {self.data_dir}")
            return False, 0, {}
        
        logger.info("\n📊 데이터셋 구조 분석:")
        logger.info("=" * 50)
        
        total_images = 0
        class_counts = {}
        
        # 유효한 디렉터리만 처리
        valid_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        
        if not valid_dirs:
            logger.error("❌ 유효한 클래스 디렉터리를 찾을 수 없습니다!")
            return False, 0, {}
        
        for class_name in sorted(valid_dirs):
            class_path = os.path.join(self.data_dir, class_name)
            valid_images = []
            
            try:
                files = os.listdir(class_path)
                for file in files:
                    if file.lower().endswith(self.supported_formats):
                        file_path = os.path.join(class_path, file)
                        if self._is_valid_image(file_path):
                            valid_images.append(file_path)
            except Exception as e:
                logger.warning(f"⚠️ 클래스 {class_name} 처리 중 오류: {e}")
                continue
            
            count = len(valid_images)
            class_counts[class_name] = count
            total_images += count
            logger.info(f"  📁 {class_name}: {count:4d}개 이미지")
        
        logger.info(f"\n📈 총 이미지 수: {total_images:,}개")
        logger.info(f"📝 클래스 수: {len(class_counts)}개")
        
        if not class_counts or total_images == 0:
            logger.error("❌ 유효한 이미지를 찾을 수 없습니다!")
            return False, 0, {}
        
        self._analyze_class_imbalance(class_counts)
        return True, len(class_counts), class_counts
    
    def _is_valid_image(self, file_path: str) -> bool:
        """이미지 유효성 검사 개선"""
        try:
            # 파일 크기 체크
            if os.path.getsize(file_path) < 1024:  # 1KB 미만은 제외
                return False
            
            # PIL로 이미지 검증
            with Image.open(file_path) as img:
                img.verify()
                
            # 다시 열어서 기본 정보 확인
            with Image.open(file_path) as img:
                width, height = img.size
                if width < 32 or height < 32:  # 너무 작은 이미지 제외
                    return False
                    
            return True
        except Exception:
            return False
    
    def _analyze_class_imbalance(self, class_counts):
        """클래스 불균형 분석"""
        counts = list(class_counts.values())
        if not counts:
            return
            
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        
        logger.info(f"\n📊 클래스별 분포:")
        logger.info(f"   최소: {min_count}개")
        logger.info(f"   최대: {max_count}개")
        logger.info(f"   평균: {avg_count:.1f}개")
        
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            logger.info(f"   불균형 비율: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 3:
                logger.warning("⚠️ 심각한 클래스 불균형 감지!")
            elif imbalance_ratio > 2:
                logger.warning("⚠️ 약간의 클래스 불균형 감지")
        
        if min_count < 10:
            logger.warning(f"⚠️ 경고: 일부 클래스의 이미지가 너무 적습니다 (최소 {min_count}개)")

# ========== 이미지 제너레이터 클래스 개선 ==============
class ImprovedImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, class_names, batch_size, 
                 target_size=(224,224), augment=False, class_weights=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.num_classes = len(class_names)
        self.samples = len(image_paths)
        self.class_weights = class_weights or self._calculate_class_weights()
        self.indices = np.arange(len(self.image_paths))
        
        # 캐시 크기 제한
        self.cache_size = min(50, len(image_paths) // 10)  # 캐시 크기 감소
        self.image_cache = {}
        
        if augment:
            np.random.shuffle(self.indices)
    
    def _calculate_class_weights(self):
        """클래스 가중치 계산"""
        try:
            unique_labels = np.unique(self.labels)
            class_weights = compute_class_weight(
                'balanced', classes=unique_labels, y=self.labels
            )
            return dict(zip(unique_labels, class_weights))
        except Exception as e:
            logger.warning(f"클래스 가중치 계산 실패: {e}")
            return {}
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            try:
                img_array = self._load_and_preprocess_image(i)
                if img_array is not None:
                    if self.augment:
                        img_array = self._augment_image(img_array)
                    
                    # 정규화는 모델에서 처리하므로 여기서는 0-255 유지
                    img_array = img_array.astype(np.float32)
                    label = tf.keras.utils.to_categorical(self.labels[i], num_classes=self.num_classes)
                    
                    batch_x.append(img_array)
                    batch_y.append(label)
                else:
                    # 실패한 경우 기본 이미지 사용
                    batch_x.append(self._get_default_image())
                    batch_y.append(tf.keras.utils.to_categorical(0, num_classes=self.num_classes))
                    
            except Exception as e:
                logger.warning(f"⚠️ 이미지 로드 실패: {self.image_paths[i]} - {e}")
                batch_x.append(self._get_default_image())
                batch_y.append(tf.keras.utils.to_categorical(0, num_classes=self.num_classes))
        
        # 배치가 비어있지 않도록 보장
        if not batch_x:
            batch_x = [self._get_default_image()]
            batch_y = [tf.keras.utils.to_categorical(0, num_classes=self.num_classes)]
        
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)
    
    def _load_and_preprocess_image(self, idx):
        """이미지 로드 및 전처리 개선"""
        image_path = self.image_paths[idx]
        cache_key = hashlib.md5(image_path.encode()).hexdigest()[:16]  # 짧은 해시
        
        # 캐시 확인
        if cache_key in self.image_cache:
            return self.image_cache[cache_key].copy()
        
        try:
            # OpenCV로 이미지 로드 시도
            img_array = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if img_array is None:
                # PIL로 대체 시도
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
            else:
                # BGR to RGB 변환
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # 리사이즈
            img_array = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_AREA)
            
            # 캐시에 저장 (제한된 수만)
            if len(self.image_cache) < self.cache_size:
                self.image_cache[cache_key] = img_array.copy()
            
            return img_array
            
        except Exception as e:
            logger.warning(f"이미지 로드 실패: {image_path} - {e}")
            return None
    
    def _get_default_image(self):
        """기본 이미지 생성"""
        return np.zeros((*self.target_size, 3), dtype=np.float32)
    
    def _augment_image(self, image):
        """이미지 증강 (가벼운 버전)"""
        image = image.copy()
        
        # 수평 뒤집기
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # 밝기 조정
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255)
        
        # 대비 조정
        if np.random.random() > 0.7:
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = np.clip((image - 128) * contrast_factor + 128, 0, 255)
        
        return image.astype(np.float32)
    
    def on_epoch_end(self):
        """에포크 종료시 셔플"""
        if self.augment:
            np.random.shuffle(self.indices)
        
        # 메모리 정리
        if len(self.image_cache) > self.cache_size * 2:
            # 캐시 절반 제거
            keys_to_remove = list(self.image_cache.keys())[::2]
            for key in keys_to_remove:
                del self.image_cache[key]
            gc.collect()

# ========== 학습 매니저 클래스 개선 ==============
class TrainingManager:
    def __init__(self, model_name: str = 'face_shape_model'):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉터리 생성"""
        directories = ['logs', 'models', 'checkpoints']
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"📁 디렉터리 생성/확인: {directory}")
            except Exception as e:
                logger.warning(f"디렉터리 생성 실패: {directory} - {e}")
    
    def compile_model(self, model: keras.Model, num_classes: int):
        """모델 컴파일 개선"""
        # 학습률 스케줄 개선
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate,
            decay_steps=1000,
            alpha=0.01  # 최소 학습률 증가
        )
        
        # 옵티마이저 설정
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Mixed Precision 최적화
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # 컴파일
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info("📊 모델 구조:")
        try:
            model.summary(print_fn=logger.info)
        except:
            logger.info("모델 구조 출력 실패")
        
        return model
    
    def get_callbacks(self):
        """콜백 설정 개선"""
        callbacks = []
        
        try:
            # 모델 체크포인트
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join('checkpoints', f'best_{self.model_name}_{{epoch:02d}}_{{val_accuracy:.4f}}.weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='max'
            )
            callbacks.append(checkpoint_callback)
            
            # 조기 종료
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # patience 감소
                restore_best_weights=True,
                verbose=1,
                mode='min'
            )
            callbacks.append(early_stopping)
            
            # 학습률 감소
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,  # patience 감소
                min_lr=1e-7,
                verbose=1,
                mode='min'
            )
            callbacks.append(reduce_lr)
            
            # CSV 로거 (간단한 경로)
            csv_logger = tf.keras.callbacks.CSVLogger(
                filename=f'training_log_{self.timestamp}.csv',
                append=True
            )
            callbacks.append(csv_logger)
            
        except Exception as e:
            logger.warning(f"일부 콜백 설정 실패: {e}")
        
        return callbacks
    
    def save_model(self, model: keras.Model):
        """모델 저장 개선"""
        saved_successfully = False
        
        try:
            # Keras 형식으로 저장
            keras_path = os.path.join('models', f'{self.model_name}.keras')
            model.save(keras_path)
            logger.info(f"✅ Keras 모델 저장됨: {keras_path}")
            saved_successfully = True
        except Exception as e:
            logger.warning(f"Keras 모델 저장 실패: {e}")
        
        try:
            # 가중치만 저장
            weights_path = os.path.join('models', f'{self.model_name}.weights.h5')
            model.save_weights(weights_path)
            logger.info(f"✅ 가중치 저장됨: {weights_path}")
            saved_successfully = True
        except Exception as e:
            logger.warning(f"가중치 저장 실패: {e}")
        
        try:
            # 모델 구조 JSON으로 저장
            model_json = model.to_json()
            json_path = os.path.join('models', f'{self.model_name}_architecture.json')
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json_file.write(model_json)
            logger.info(f"✅ 모델 구조 저장됨: {json_path}")
        except Exception as e:
            logger.warning(f"모델 구조 저장 실패: {e}")
        
        if not saved_successfully:
            logger.error("❌ 모든 모델 저장 방법이 실패했습니다!")
    
    def zip_results(self):
        """결과 압축"""
        try:
            output_zip = f'result_package_{self.timestamp}'
            if os.path.exists('models') and os.listdir('models'):
                shutil.make_archive(base_name=output_zip, format='zip', root_dir='models')
                logger.info(f"📦 결과 ZIP 압축 완료: {output_zip}.zip")
            else:
                logger.warning("압축할 모델 파일이 없습니다.")
        except Exception as e:
            logger.error(f"ZIP 압축 실패: {e}")

# ========== 데이터 생성기 함수 개선 ==============
def create_data_generators(data_dir, batch_size=16, validation_split=0.2, target_size=(224,224)):
    """데이터 제너레이터 생성 함수 개선"""
    logger.info(f"📂 데이터 디렉터리: {data_dir}")
    
    data_manager = DataManager(data_dir)
    is_valid, num_classes, class_counts = data_manager.validate_data_structure()
    
    if not is_valid:
        raise ValueError("데이터 구조가 유효하지 않습니다.")
    
    # 이미지 경로와 레이블 수집
    image_paths, labels, class_names = [], [], list(class_counts.keys())
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        class_images = []
        
        try:
            for file in os.listdir(class_path):
                if file.lower().endswith(data_manager.supported_formats):
                    file_path = os.path.join(class_path, file)
                    if data_manager._is_valid_image(file_path):
                        class_images.append(file_path)
            
            # 클래스별 최대 이미지 수 제한 (메모리 관리)
            if len(class_images) > 1000:
                class_images = np.random.choice(class_images, 1000, replace=False).tolist()
                logger.info(f"⚠️ {class_name} 클래스: {len(class_images)}개로 제한")
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            
        except Exception as e:
            logger.error(f"클래스 {class_name} 처리 중 오류: {e}")
            continue
    
    if not image_paths:
        raise ValueError("유효한 이미지를 찾을 수 없습니다.")
    
    logger.info(f"📊 총 수집된 이미지: {len(image_paths)}개")
    
    # 클래스 가중치 계산
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = dict(zip(np.unique(labels), class_weights))
    except Exception as e:
        logger.warning(f"클래스 가중치 계산 실패: {e}")
        class_weight_dict = {}
    
    # 훈련/검증 분할
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=validation_split, 
            stratify=labels, 
            random_state=42
        )
    except Exception as e:
        logger.warning(f"Stratified split 실패, 일반 split 사용: {e}")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=validation_split, 
            random_state=42
        )
    
    logger.info(f"📈 학습 샘플: {len(train_paths)}개")
    logger.info(f"📊 검증 샘플: {len(val_paths)}개")
    
        # 제너레이터 생성
    train_gen = ImprovedImageGenerator(
        train_paths, train_labels, class_names, batch_size,
        target_size=target_size, augment=True, class_weights=class_weight_dict
    )
    val_gen = ImprovedImageGenerator(
        val_paths, val_labels, class_names, batch_size,
        target_size=target_size, augment=False, class_weights=class_weight_dict
    )

    return train_gen, val_gen, class_names, class_weight_dict

if __name__ == "__main__":
    # ====== 사용자 설정 ======
    DATA_DIR = "./dataset/train"  # 데이터셋 경로 (train 하위 폴더로 수정)
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    TARGET_SIZE = (224, 224)
    MODEL_NAME = "face_shape_model"
    EPOCHS = 30

    logger.info("\n🚀 [MAIN] 학습 파이프라인 시작!")

    # 1. GPU 최적화
    GPUOptimizer.setup_gpu_optimized(memory_limit=8192)
    GPUOptimizer.check_tensorflow_gpu()

    # 2. 데이터 생성기 준비
    train_gen, val_gen, class_names, class_weight_dict = create_data_generators(
        DATA_DIR, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, target_size=TARGET_SIZE
    )
    num_classes = len(class_names)

    # 3. 모델 생성 및 컴파일
    model = ModelBuilder.create_efficient_model(input_shape=(*TARGET_SIZE, 3), num_classes=num_classes)
    trainer = TrainingManager(model_name=MODEL_NAME)
    model = trainer.compile_model(model, num_classes=num_classes)

    # 4. 콜백 준비
    callbacks = trainer.get_callbacks()

    # 5. 모델 학습
    logger.info("\n🟢 모델 학습 시작!")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # 6. 모델 저장
    trainer.save_model(model)
    trainer.zip_results()

    logger.info("\n✅ [MAIN] 학습 파이프라인 완료!")
