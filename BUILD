CC('gcc')

CXX('g++')

# PROTOC('protoc')

CFLAGS('-g -pipe -Wall -std=c99')

CXXFLAGS('-g -pipe -Wall -std=c++17')

LDFLAGS('-L.')

LDLIBS('-lpthread')
LDLIBS('-lprotobuf')

BINARY(
    name = 'cola',
    includes = 'src/',
    sources = [
        'src/cola/*.cc',
        'src/cola/base/*.cc',
        'src/cola/core/*.cc',
        'src/cola/data/*.cc',
        'src/cola/layers/*.cc',
        'src/cola/optimizers/*.cc',
    ],
    protos = ['src/cola/proto/*.proto']
)

TEST(
    name='test',
    includes = 'src/',
    sources=[
        'src/test/*.cc',
        'src/cola/base/*.cc',
        'src/cola/core/*.cc',
        'src/cola/data/*.cc',
        'src/cola/layers/*.cc',
        'src/cola/optimizers/*.cc',
    ],
    protos = ['src/cola/proto/*.proto']
)
