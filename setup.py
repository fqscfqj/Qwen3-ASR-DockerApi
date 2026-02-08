from setuptools import setup, find_packages

setup(
    name='qwen3-asr-toolkit',
    version='1.0.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'dashscope',
        'librosa',
        'soundfile',
        'silero_vad',
        'pydub',
        'tqdm',
        'numpy==1.26.4',
        'srt'
    ],
    extras_require={
        'server': [
            'fastapi==0.115.2',
            'uvicorn==0.30.6',
            'python-multipart==0.0.22',
            'transformers==4.48.1',
            'torch==2.6.0'
        ],
    },
    entry_points={
        'console_scripts': [
            'qwen3-asr=qwen3_asr_toolkit.call_api:main',
            'qwen3-asr-server=qwen3_asr_toolkit.server:main'
        ]
    },
    author='He Wang',
    author_email='hwang2001@mail.nwpu.edu.cn',
    description='Python toolkit for the Qwen3-ASR API—parallel high‑throughput calls, robust long‑audio transcription, multi‑sample‑rate support.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/QwenLM/Qwen3-ASR-Toolkit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
