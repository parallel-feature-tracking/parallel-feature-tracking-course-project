#include <iostream>

#ifndef CONSTS_H
#define CONSTS_H

#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <list>

const float DIST_THRESHOLD = .5;
const float OPACITY_THRESHOLD  = .1;

const int MIN_NUM_VOXEL_IN_FEATURE = 10;
const int FT_DIRECT = 0;
const int FT_LINEAR = 1;
const int FT_POLYNO = 2;
const int FT_FORWARD  = 0;
const int FT_BACKWARD = 1;
const int DEFAULT_TF_RES = 1024;

// Surfaces
const int SURFACE_NULL   = -1;  // default
const int LEFT   = 0;   // x = 0
const int RIGHT  = 1;   // x = xs-1
const int BOTTOM = 2;   // y = 0
const int TOP    = 3;   // y = ys-1
const int FRONT  = 4;   // z = 0
const int BACK   = 5;   // z = zs-1

using namespace std;

namespace util {
template<class T>
class vec3 {
public:
    T x, y, z;
    vec3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
    T*    data()                              { return &x; }
    T     volumeSize()                        { return x * y * z; }
    float magnituteSquared()                  { return x*x + y*y + z*z; }
    float magnitute()                         { return sqrt((*this).magnituteSquared()); }
    float distanceFrom(vec3 const& rhs) const { return (*this - rhs).magnitute(); }
    vec3  operator -  ()                      { return vec3(-x, -y, -z); }
    vec3  operator +  (vec3 const& rhs) const { vec3 t(*this); t+=rhs; return t; }
    vec3  operator -  (vec3 const& rhs) const { vec3 t(*this); t-=rhs; return t; }
    vec3  operator *  (vec3 const& rhs) const { vec3 t(*this); t*=rhs; return t; }
    vec3  operator /  (vec3 const& rhs) const { vec3 t(*this); t/=rhs; return t; }
    vec3  operator *  (float scale)     const { vec3 t(*this); t*=scale; return t; }
    vec3  operator /  (float scale)     const { vec3 t(*this); t/=scale; return t; }
    vec3& operator += (vec3 const& rhs)       { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
    vec3& operator -= (vec3 const& rhs)       { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
    vec3& operator *= (vec3 const& rhs)       { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    vec3& operator /= (vec3 const& rhs)       { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    vec3& operator *= (float scale)           { x*=scale, y*=scale, z*=scale; return *this; }
    vec3& operator /= (float scale)           { x/=scale, y/=scale, z/=scale; return *this; }
    bool  operator == (vec3 const& rhs) const { return x==rhs.x && y==rhs.y && z==rhs.z; }
    bool  operator != (vec3 const& rhs) const { return !(*this == rhs); }
};

static inline string ltrim(const string &s) {    // trim string from left
    int start = s.find_first_not_of(' ');
    return s.substr(start, s.size() - start);
}

static inline string rtrim(const string &s) {    // trim string from right
    return s.substr(0, s.find_last_not_of(' ')+1);
}

static inline string trim(const string &s) {     // trim all white spaces
    return ltrim(rtrim(s));
}

static inline bool ascending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
    return lhs.second < rhs.second;
}

static inline bool descending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
    return !ascending(lhs, rhs);
}

static inline int round(float f) {
    return static_cast<int>(floor(f + 0.5f));
}

static inline vec3<int> min(const vec3<int>& v1, const vec3<int>& v2) {
    return vec3<int>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
}

static inline vec3<int> max(const vec3<int>& v1, const vec3<int>& v2) {
    return vec3<int>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
}
}

typedef util::vec3<int> vec3i;

struct Leaf {
int id;         // unique id for each leaf
int root;       // id of the residing block
int tip;        // id of the adjacent block
vec3i centroid; // centroid of boundary surface

bool operator==(const Leaf& rhs) const {
    return (*this).id==rhs.id && (*this).root==rhs.root && (*this).tip== rhs.tip;
}
};

struct Feature {
int   id;        // Unique ID for each feature
int   maskValue; // color id of the feature
vec3i ctr;       // Centroid position of the feature
vec3i min;       // Minimum position (x, y, z) of the feature
vec3i max;       // Maximum position (x, y, z) of the feature
std::array<vec3i, 6> boundaryCtr; // center point on each boundary surface
std::array<vec3i, 6> boundaryMin; // min value on each boundary surface
std::array<vec3i, 6> boundaryMax; // max value on each boundary surface
std::array<int, 6> numVoxelOnSurface; // number of voxels on each boundary surface
std::list<vec3i> edgeVoxels; // Edge information of the feature
std::list<vec3i> bodyVoxels; // All the voxels in the feature
std::vector<int> touchedSurfaces;
};

#endif // CONSTS_H

#ifndef METADATA_H
#define METADATA_H

class Metadata {
public:
    int      start()      const { return start_; }
    int      end()        const { return end_; }
    string   prefix()     const { return prefix_; }
    string   suffix()     const { return suffix_; }
    string   path()       const { return path_; }
    string   tfPath()     const { return tfPath_; }
    string   timeFormat() const { return timeFormat_; }
    vec3i    volumeDim()  const { return volumeDim_; }

    Metadata(const std::string &fpath);
   ~Metadata();

private:
    int      start_;
    int      end_;
    string   prefix_;
    string   suffix_;
    string   path_;
    string   tfPath_;
    string   timeFormat_;
    vec3i    volumeDim_;
};

#endif // METADATA_H

#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

class FeatureTracker {

public:
    FeatureTracker(vec3i dim);
   ~FeatureTracker();

    void ExtractAllFeatures();
    void FindNewFeature(vec3i seed);
    void TrackFeature(float* pData, int direction, int mode);

    void SaveExtractedFeatures(int index)   { featureSequence[index] = currentFeatures; }
    void SetDataPtr(float* pData)           { data.assign(pData, pData+volumeSize); }
    void SetTFRes(int res)                  { tfRes = res; }
    void SetTFMap(float* map)               { tfMap.assign(map, map+tfRes); }

    float* GetMaskPtr()                     { return mask.data(); }
    int GetTFResolution()             const { return tfRes; }
    int GetVoxelIndex(const vec3i& v) const { return blockDim.x*blockDim.y*v.z+blockDim.x*v.y+v.x; }
    std::vector<Feature> GetFeatures(int t) { return featureSequence[t]; }

private:
    float getOpacity(float value) { return tfMap[static_cast<int>(value * (tfRes-1))]; }
    vec3i predictRegion(int index, int direction, int mode); // Predict region t based on direction, returns offset
    void fillRegion(Feature& f, const vec3i& offset);        // Scanline algorithm - fills everything inside edge
    void expandRegion(Feature& f);                           // Grows edge where possible
    void shrinkRegion(Feature& f);                           // Shrinks edge where nescessary
    bool expandEdge(Feature& f, const vec3i& voxel);         // Sub-func inside expandRegion
    void shrinkEdge(Feature& f, const vec3i& voxel);         // Sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);                   // Update the feature vectors information after tracking
    void updateFeatureBoundary(Feature& f, const vec3i& voxel, int surface);
    Feature createNewFeature();
    
    int tfRes;              // Default transfer function resolution
    int volumeSize;
    int timeLeft2Forward;
    int timeLeft2Backward;
    float maskValue;  // Global mask value for newly detected features

    vec3i blockDim;

    std::array<vec3i, 6> boundaryCtr;  // centroid of the boundary surface
    std::array<vec3i, 6> boundaryMin;  // min values of the boundary surface
    std::array<vec3i, 6> boundaryMax;  // max values of the boundary surface

    std::vector<float> data;        // Raw volume intensity value
    std::vector<float> mask;        // Mask volume, same size with a time step data
    std::vector<float> maskPrev;    // Mask volume, same size with a time step data
    std::vector<float> tfMap;       // Tranfer function setting
    std::vector<Feature> currentFeatures; // Features info in current time step
    std::vector<Feature> backup1Features; // ... in the 1st backup time step
    std::vector<Feature> backup2Features; // ... in the 2nd backup time step
    std::vector<Feature> backup3Features; // ... in the 3rd backup time step

    std::unordered_map<int, std::vector<Feature> > featureSequence;
};

#endif // FEATURETRACKER_H

#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>

class DataManager {

public:
    DataManager(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx);
   ~DataManager();

    float* GetDataPtr(int t) { return dataSequence[t].data(); }
    float* GetTFMap()        { return tfMap.data(); }
    int GetTFRes()           { return tfRes > 0 ? tfRes : DEFAULT_TF_RES; }
    vec3i GetBlockDim()      { return blockDim; }

    void LoadDataSequence(const Metadata& meta, const int currentT);
    void SaveMaskVolume(float *pData, const Metadata& meta, const int t);
    void SaveMaskVolumeMpi(float *pData, const Metadata& meta, const int t);

private:
    vec3i blockDim;
    int volumeSize;
    int tfRes;
    
    MPI_Datatype fileType;

    std::unordered_map<int, std::vector<float> > dataSequence;
    std::vector<float> outputData;
    std::vector<float> tfMap;

    void preprocessData(std::vector<float>& data);
};

#endif // DATAMANAGER_H

#ifndef BLOCKCONTROLLER_H
#define BLOCKCONTROLLER_H

class BlockController {

public:
    BlockController();
   ~BlockController();

    void InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx);
    void TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx);
    void SetCurrentTimestep(int t) { currentT = t; }

    std::vector<int> GetAdjacentBlockIds();
    std::vector<Leaf> GetConnectivityTree() { return connectivityTree; }
    void SetConnectivityTree(const std::vector<Leaf>& tree) { connectivityTree = tree; }
    void UpdateConnectivityTree(int currentBlockId, const vec3i& blockIdx);

private:
    DataManager    *pDataManager;
    FeatureTracker *pFeatureTracker;
    vec3i           blockDim;
    int             currentT;
    
    std::unordered_map<int, int> adjacentBlocks;
    std::vector<Leaf> connectivityTree;

    void initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx);
};

#endif // BLOCKCONTROLLER_H

#ifndef MPICONTROLLER_H
#define MPICONTROLLER_H

#include <fstream>
#include <mpi.h>
// #include "Utils.h"

class MpiController {
public:
MpiController();
~MpiController();

void InitWith(int argc, char** argv);
void Start();

private:
BlockController *pBlockController;
Metadata *pMetadata;

MPI_Datatype MPI_TYPE_LEAF;

int myRank;
int numProc;
int currentT;  // current timestep

vec3i gridDim;  // #processes in each dimension (xyz)
vec3i blockIdx; // xyz coordinate of current processor

// for global graph
void gatherLeaves();

// for feature graph
std::vector<int> adjacentBlocks;
void syncLeaves();
void updateFeatureTable(const Leaf& leaf);
bool toSend, toRecv;
bool anySend, anyRecv;

// global feature info
typedef std::unordered_map<int, std::vector<int> > FeatureTable;
FeatureTable featureTable;
std::unordered_map<int, FeatureTable> featureTableVector; // for time varying data

void mergeCorrespondingEdges(std::vector<Leaf> leaves);
};

#endif // MPICONTROLLER_H


// #region MPIController
MpiController::MpiController() {}

MpiController::~MpiController() {
pBlockController->~BlockController();
pMetadata->~Metadata();
MPI_Finalize();
}

void MpiController::InitWith(int argc, char **argv) {
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
MPI_Comm_size(MPI_COMM_WORLD, &numProc);

// declare new type for edge
MPI_Type_contiguous(sizeof(Leaf), MPI_BYTE, &MPI_TYPE_LEAF);
MPI_Type_commit(&MPI_TYPE_LEAF);

gridDim = vec3i(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

blockIdx.z = myRank / (gridDim.x * gridDim.y);
blockIdx.y = (myRank - blockIdx.z * gridDim.x * gridDim.y) / gridDim.x;
blockIdx.x = myRank % gridDim.x;

std::string metaPath(argv[4]);
pMetadata = new Metadata(metaPath);

currentT = pMetadata->start();

pBlockController = new BlockController();
pBlockController->SetCurrentTimestep(currentT);
pBlockController->InitParameters(*pMetadata, gridDim, blockIdx);
}

void MpiController::Start() {
while (currentT < pMetadata->end()) {
    currentT++;
    if (myRank == 0) {
        std::cout << "----- " << currentT << " -----" << std::endl;    
    }

    pBlockController->SetCurrentTimestep(currentT);
    pBlockController->TrackForward(*pMetadata, gridDim, blockIdx);
    pBlockController->UpdateConnectivityTree(myRank, blockIdx);

    featureTable.clear();

    adjacentBlocks = pBlockController->GetAdjacentBlockIds();
    toSend = toRecv = anySend = anyRecv = true;
    while (anySend || anyRecv) {
        syncLeaves();
    }

    // gatherLeaves();

    if (myRank == 0) {
        for (const auto& f : featureTable) {
            std::cout << "["<<myRank<<"] " << f.first << ":";
            for (const auto& id : f.second) {
                std::cout << id << " ";
            }
            std::cout << std::endl;
        }            
    }

    featureTableVector[currentT] = featureTable;
}

if (myRank == 0) {
    std::cout << "all jobs done." << std::endl;
}
}

void MpiController::gatherLeaves() {
std::vector<Leaf> leaves = pBlockController->GetConnectivityTree();
int numLeaves = leaves.size();

// a vector that holds the number of leaves for each block
std::vector<int> numLeavesGlobal(numProc);
MPI_Allgather(&numLeaves, 1, MPI_INT, numLeavesGlobal.data(), 1, MPI_INT, MPI_COMM_WORLD);

// be careful, the last argument of std::accumulate controls both value and type
int numLeavesSum = std::accumulate(numLeavesGlobal.begin(), numLeavesGlobal.end(), 0);
std::vector<Leaf> leavesGlobal(numLeavesSum);

// where leaves received from other blocks should be places. 
std::vector<int> displacements(numProc, 0);
for (int i = 1; i < numProc; ++i) {
    displacements[i] = numLeavesGlobal[i-1] + displacements[i-1];
}

// gather leaves from all blocks
MPI_Allgatherv(leaves.data(), numLeaves, MPI_TYPE_LEAF, leavesGlobal.data(),
    numLeavesGlobal.data(), displacements.data(), MPI_TYPE_LEAF, MPI_COMM_WORLD);

mergeCorrespondingEdges(leavesGlobal);
}

void MpiController::syncLeaves() {
std::vector<Leaf> myLeaves = pBlockController->GetConnectivityTree();
int numLeaves = myLeaves.size();

int recvId = toRecv ? myRank : -1;
std::vector<int> blocksNeedRecv(numProc, -1);   
MPI_Allgather(&recvId, 1, MPI_INT, blocksNeedRecv.data(), 1, MPI_INT, MPI_COMM_WORLD);

std::vector<int> adjacentBlocksNeedRecv;

// find if any of my neighbors need to receive
std::sort(adjacentBlocks.begin(), adjacentBlocks.end());
std::sort(blocksNeedRecv.begin(), blocksNeedRecv.end());
std::set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(),
    blocksNeedRecv.begin(), blocksNeedRecv.end(), back_inserter(adjacentBlocksNeedRecv));

// no bother to send if nobody need to receive
toSend = adjacentBlocksNeedRecv.empty() ? false : true;

if (toRecv) {
    for (auto neighbor : adjacentBlocks) {
        int numLeaves = 0;
        MPI_Request request;
        // 1. see how many leaves my neighbor have
        MPI_Irecv(&numLeaves, 1, MPI_INT, neighbor, 100, MPI_COMM_WORLD, &request);
        // 2. if they have any, get them
        if (numLeaves != 0) {
            std::vector<Leaf> leaves(numLeaves);
            MPI_Irecv(leaves.data(), numLeaves, MPI_TYPE_LEAF, neighbor, 101, MPI_COMM_WORLD, &request);
            for (const auto& leaf : leaves) {
                // 3. if I don't previously have the leaf my neighbor send to me, take it
                if (std::find(myLeaves.begin(), myLeaves.end(), leaf) == myLeaves.end()) {
                    myLeaves.push_back(leaf);
                }
            }
        }
    }
    toRecv = false;
}

if (toSend) {
    for (auto neighbor : adjacentBlocksNeedRecv) {
        // 1. tell them how many leaves I have, even if I have none
        MPI_Send(&numLeaves, 1, MPI_INT, neighbor, 100, MPI_COMM_WORLD);
        // 2. if I have any, send them
        if (numLeaves > 0) {
            MPI_Send(myLeaves.data(), numLeaves, MPI_TYPE_LEAF, neighbor, 101, MPI_COMM_WORLD);
        }
    }
}

mergeCorrespondingEdges(myLeaves);
pBlockController->SetConnectivityTree(myLeaves);

// anySend = any(toSend), anyRecv = any(toRecv)
MPI_Allreduce(&toSend, &anySend, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
MPI_Allreduce(&toRecv, &anyRecv, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

void MpiController::mergeCorrespondingEdges(std::vector<Leaf> leaves) {
for (auto i = 0; i < leaves.size(); ++i) {
    Leaf& p = leaves[i];
    for (auto j = i+1; j < leaves.size(); ++j) {
        Leaf& q = leaves[j];

        // sync leaf id of feature with the smaller one if two match
        if (p.root == q.tip && p.tip == q.root && (p.root == myRank || p.tip == myRank) &&
            p.centroid.distanceFrom(q.centroid) < DIST_THRESHOLD) {
            p.id = q.id = std::min(p.id, q.id);
            // updateFeatureTable(leaves[i]);
        }
    }
}

// if either root or tip equals to myRank, add to featureTable
// if both root and tip are not equal to myRank,
// but the id is already in the feature table, update featureTable
for (const auto& leaf : leaves) {
    if (leaf.root == myRank || leaf.tip == myRank || featureTable.find(leaf.id) != featureTable.end()) {
        updateFeatureTable(leaf);
    }
}
}

void MpiController::updateFeatureTable(const Leaf& leaf) {
if (featureTable.find(leaf.id) == featureTable.end()) {
    std::vector<int> values;
    values.push_back(leaf.root);
    values.push_back(leaf.tip);
    featureTable[leaf.id] = values;
    toRecv = true;
} else {
    std::vector<int> &value = featureTable[leaf.id];
    if (std::find(value.begin(), value.end(), leaf.root) == value.end()) {
        value.push_back(leaf.root);
        toRecv = true;
    }
    if (std::find(value.begin(), value.end(), leaf.tip) == value.end()) {
        value.push_back(leaf.tip);
        toRecv = true;
    }
}
}

// #endregion MPIController

Metadata::Metadata(const string &fpath) {
    ifstream meta(fpath.c_str());
    if (!meta) {
        std::cout << "cannot read meta file: " << fpath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (getline(meta, line)) {
        size_t pos = line.find('=');
        if (pos == line.npos) { continue; }

        std::string value = util::trim(line.substr(pos+1));
        if (line.find("start") != line.npos) {
            start_ = atoi(value.c_str());
        } else if (line.find("end") != line.npos) {
            end_ = atoi(value.c_str());
        } else {
            // remove leading & trailing chars () or ""
            value = value.substr(1, value.size()-2);

            if (line.find("prefix") != line.npos) {
                prefix_ = value;
            } else if (line.find("suffix") != line.npos) {
                suffix_ = value;
            } else if (line.find("path") != line.npos) {
                path_ = value;
            } else if (line.find("tfPath") != line.npos) {
                tfPath_ = value;
            } else if (line.find("timeFormat") != line.npos) {
                timeFormat_ = value;
            } else if (line.find("volumeDim") != line.npos) {
                std::vector<int> dim;
                size_t pos = 0;
                while ((pos = value.find(',')) != value.npos) {
                    dim.push_back(atoi(util::trim(value.substr(0, pos)).c_str()));
                    value.erase(0, pos+1);
                }
                dim.push_back(atoi(util::trim(value).c_str()));

                if (dim.size() != 3) {
                    std::cout << "incorrect volumeDim format" << std::endl;
                    exit(EXIT_FAILURE);
                }
                volumeDim_ = vec3i(dim[0], dim[1], dim[2]);
            }
        }
    }
}

Metadata::~Metadata() {}

BlockController::BlockController() {}

BlockController::~BlockController() {
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    initAdjacentBlocks(gridDim, blockIdx);
    
    pDataManager = new DataManager(meta, gridDim, blockIdx);
    pDataManager->LoadDataSequence(meta, currentT);

    pFeatureTracker = new FeatureTracker(pDataManager->GetBlockDim());
    pFeatureTracker->SetTFRes(pDataManager->GetTFRes());
    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->SetDataPtr(pDataManager->GetDataPtr(currentT));
}

void BlockController::TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    pDataManager->LoadDataSequence(meta, currentT);

    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->ExtractAllFeatures();
    pFeatureTracker->TrackFeature(pDataManager->GetDataPtr(currentT), FT_FORWARD, FT_DIRECT);
    pFeatureTracker->SaveExtractedFeatures(currentT);

    pDataManager->SaveMaskVolume(pFeatureTracker->GetMaskPtr(), meta, currentT);
    // pDataManager->SaveMaskVolumeMpi(pFeatureTracker->GetMaskPtr(), meta, currentT);
}

void BlockController::initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx) {
    int px = gridDim.x,   py = gridDim.y,   pz = gridDim.z;
    int x = blockIdx.x,   y = blockIdx.y,   z = blockIdx.z;

    adjacentBlocks[LEFT]   = x-1 >= 0  ? px*py*z + px*y + x - 1 : -1;
    adjacentBlocks[RIGHT]  = x+1 <  px ? px*py*z + px*y + x + 1 : -1;
    adjacentBlocks[BOTTOM] = y-1 >= 0  ? px*py*z + px*(y-1) + x : -1;
    adjacentBlocks[TOP]    = y+1 <  py ? px*py*z + px*(y+1) + x : -1;
    adjacentBlocks[FRONT]  = z-1 >= 0  ? px*py*(z-1) + px*y + x : -1;
    adjacentBlocks[BACK]   = z+1 <  pz ? px*py*(z+1) + px*y + x : -1;
}

std::vector<int> BlockController::GetAdjacentBlockIds() {
    std::vector<int> indices;
    for (auto i = 0; i < adjacentBlocks.size(); ++i) {
        if (adjacentBlocks[i] != -1) {
            indices.push_back(adjacentBlocks[i]);
        }
    }
    return indices;
}

void BlockController::UpdateConnectivityTree(int currentBlockId, const vec3i& blockIdx) {
    connectivityTree.clear();

    std::vector<Feature> features = pFeatureTracker->GetFeatures(currentT);
    for (const auto& f : features) {
        for (auto surface : f.touchedSurfaces) {
            int adjacentBlockId = adjacentBlocks[surface];
            if (adjacentBlockId == -1) {
                continue;
            }

            Leaf leaf;
            leaf.id       = f.id;
            leaf.root     = currentBlockId;
            leaf.tip      = adjacentBlockId;
            leaf.centroid = f.boundaryCtr[surface] + blockDim * blockIdx;

            connectivityTree.push_back(leaf);
        }
    }
}


DataManager::DataManager(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    // 1. init transfer function setting
    ifstream inf(meta.tfPath().c_str(), ios::binary);
    if (!inf) {
        std::cout << "cannot load tf setting: " << meta.tfPath() << std::endl;
        exit(EXIT_FAILURE);
    }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    cout << "tfResF obtained : " << tfResF << endl;
    if (tfResF < 1) {
        std::cout << "tfResolution = " << tfResF << std::endl;
        exit(EXIT_FAILURE);
    }

    tfRes = static_cast<int>(tfResF);
    cout << "tfRes obtained : " << tfRes << endl;
    tfMap.resize(tfRes);
    inf.read(reinterpret_cast<char*>(tfMap.data()), tfRes*sizeof(float));
    inf.close();

    // 2. init parallel io parameters
    blockDim = meta.volumeDim() / gridDim;
    volumeSize = blockDim.volumeSize();

    int *gsizes = meta.volumeDim().data();
    int *subsizes = blockDim.data();
    int *starts = (blockDim * blockIdx).data();

    MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_FLOAT, &fileType);
    MPI_Type_commit(&fileType);
}

DataManager::~DataManager() {
    // cout << "Initialization of DataManager ... \n";
    MPI_Type_free(&fileType);
    // cout << "Initialization Complete of DataManager ... \n";
}

void DataManager::SaveMaskVolume(float* pData, const Metadata &meta, const int t) {
    char timestamp[21];  // up to 64-bit number
    sprintf(timestamp, (meta.timeFormat()).c_str(), t);
    int _rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + "_" + to_string(_rank) + "_"+ ".mask";
    ofstream outf(fpath.c_str(), ios::binary);
    if (!outf) {
        cerr << "cannot output to file: " << fpath.c_str() << endl;
        exit(EXIT_FAILURE);
    }

    outf.write(reinterpret_cast<char*>(pData), volumeSize*sizeof(float));
    outf.close();
    // if(_rank == 0){
    string bfpath = meta.path() + "/" + meta.prefix() + timestamp + "haha.mask";
    ofstream fout(bfpath.c_str(), ios::binary);

    for(int i=0; i<outputData.size(); i++){
        int tfindex = (int)(outputData[i] * (tfRes-1));
        outputData[i] = tfMap[tfindex];
    }

    fout.write(reinterpret_cast<char*>(outputData.data()), outputData.size()*sizeof(float));
    cout << "data written successfullly \n" << endl;
    fout.close();
    // }
    // std::cout << "mask volume created: " << fpath << std::endl;
}

void DataManager::SaveMaskVolumeMpi(float *pData, const Metadata& meta, const int t) {
    // 1. generate output file name
    char timestamp[21];  // up to 64-bit number
    sprintf(timestamp, (meta.timeFormat()).c_str(), t);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    char *cfpath = const_cast<char*>(fpath.c_str());

    // 2. parallel output file
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, cfpath, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0, MPI_FLOAT, fileType, "native", MPI_INFO_NULL);
    MPI_File_write_all(file, dataSequence[t].data(), volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // std::cout << "mask volume created: " << fpath << std::endl;
}


    // printf("[DataManager::LoadDataSequence] before the for loop\n");
void DataManager::LoadDataSequence(const Metadata& meta, const int currentT) {
    int rank_curr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_curr);
    // cout << "[LoadDataSequence] " << "started for = " << rank_curr << endl;
    // cout << "[rank: ]" << rank_curr << "the size of the dataSequence is : " << dataSequence.size() << endl;
    int _some_random_counter = 1;
    for (auto& data : dataSequence) {
        _some_random_counter++;
        // cout << "counter = " << _some_random_counter << endl;
        // cout << "data.first = " << typeid(data.first).name() << " data.second = " << typeid(data.second).name() << endl;
        try{
            // cout << "currentT = " << currentT << endl;
            // cout << "data.first = " << data.first << "  " << endl;
            // cout << "data.second size : " << data.second.size() << endl;
            // if (data.first < currentT-2 || data.first > currentT+2) {
            //     auto it = dataSequence.find(data.first);
            //     if (it != dataSequence.end()){
            //         dataSequence.erase(data.first);
            //     }else{
            //         cout << "[currentT:] " << currentT << "[data.first:] " << data.first << "data.first doesn't exist for this one....\n"; 
            //     }
            // }
        }catch(const std::exception& e){
            cout << "some exception with the reading of data :(" << endl;
            // cout << "[ERROR : ]" << e << endl;
        }
        // for(auto& val: data.second){
        //     cout << "--" << val << endl;
        // }
    }
    // cout << "the volume size if : " << volumeSize << endl;
    // printf("[DataManager::LoadDataSequence] after the for loop\n");

    for (int t = currentT-2; t <= currentT+2; ++t) {
        if (t < meta.start() || t > meta.end() || !dataSequence[t].empty()) {
            continue;
        }
    // printf("[DataManager::LoadDataSequence] after the 2nd for loop\n");

        // 1. resize to allocate buffer
        // cout << "here :)\n";
        cout << "[rank]" << rank_curr <<"failure point check ?" ;
        dataSequence[t].resize(volumeSize);
        cout << "didn't fail \n";
    // printf("[DataManager::LoadDataSequence] after dataSequence[t].resize(volumeSize)\n");

        // 2. generate file name by time step
        char timestamp[21];  // up to 64-bit number
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        // cout << "the timestamp : %d\n" << timestamp << endl;
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();
        char *cfpath = const_cast<char*>(fpath.c_str());
        
    // printf("[DataManager::LoadDataSequence] 2. generate file name by time step\n");
        // 3. parallel input file
        MPI_File file;
        int notExist = MPI_File_open(MPI_COMM_WORLD, cfpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        if (notExist) {
            std::cout << "cannot read file: " + fpath << std::endl;
            exit(EXIT_FAILURE);
        }
    // printf("[DataManager::LoadDataSequence] 3. parallet input file\n");
        int _rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        std::int64_t localVolumeSize = volumeSize;
        std::int64_t offset = localVolumeSize * _rank * sizeof(float); // Calculate the offset for each process
        MPI_File_set_view(file, offset, MPI_FLOAT, fileType, "native", MPI_INFO_NULL);
        std::cout << "Process " << _rank << " reading from the file = " << fpath 
          << " from offset = " << offset 
          << " the local volumeSize to read is = " << localVolumeSize << std::endl;
        MPI_File_read_all(file, dataSequence[t].data(), volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        cout << "the size of the datasequence loaded = " << dataSequence[t].size() << endl;
        // cout << 
        MPI_File_close(&file);
        std::ifstream bfile(cfpath, std::ios::binary | std::ios::ate);
        if (!bfile.is_open()) {
            std::cerr << "Could not open the file for reading." << std::endl;
            // return 1;
        }

        // Get the size of the file
        std::streamsize size = bfile.tellg();
        bfile.seekg(0, std::ios::beg);

        // Calculate the number of elements
        std::streamsize num_elements = size / sizeof(float);

        // Allocate memory for the float array
        outputData.resize(num_elements);

        // Read the contents of the file into the buffer
        if (bfile.read(reinterpret_cast<char*>(outputData.data()), size)) {
            std::cout << "File read successfully." << std::endl;
        } else {
            std::cerr << "Error occurred while reading the file." << std::endl;
            // return 1;
        }
        cout << "after successfully reading the file, it's size is : " << outputData.size() << endl;
        bfile.close();
        // 3. normalize data - parallel

        cout << "[rank]" << rank_curr <<"failure point check 2 ?" ;
        preprocessData(dataSequence[t]);
        cout << "[rank]" << rank_curr<<"didn't fail 2 \n";
        // cout << "[LoadDataSequence] " << "done for rank = " << rank_curr << endl;
    // printf("[DataManager::LoadDataSequence] 3. normalize data - parallel\n");
    }
}

void DataManager::preprocessData(std::vector<float>& data) {
    float min = data[0], max = data[0];
    for (int i = 1; i < volumeSize; ++i) {
        min = std::min(min, data[i]);
        max = std::max(max, data[i]);
    }

    MPI_Allreduce(&min, &min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max, &max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    for (int i = 0; i < volumeSize; ++i) {
        data[i] = (data[i] - min) / (max - min);
    }
}


FeatureTracker::FeatureTracker(vec3i dim) : blockDim(dim), maskValue(0), tfRes(1024) {
    volumeSize = blockDim.volumeSize();
    mask = std::vector<float>(volumeSize);
    maskPrev = std::vector<float>(volumeSize);
}

FeatureTracker::~FeatureTracker() {}

void FeatureTracker::ExtractAllFeatures() {
    for (int z = 0; z < blockDim.z; ++z) {
        for (int y = 0; y < blockDim.y; ++y) {
            for (int x = 0; x < blockDim.x; ++x) {
                int index = GetVoxelIndex(vec3i(x, y, z));
                if (mask[index] > 0) { 
                    continue; // point already within a feature
                }
                int tfindex = (int)(data[index] * (tfRes-1));
                if (tfMap[tfindex] >= OPACITY_THRESHOLD) {
                    FindNewFeature(vec3i(x,y,z));
                }
            }
        }
    }
}

Feature FeatureTracker::createNewFeature() {
    Feature f;
    f.id        = -1;
    f.maskValue = -1;
    f.ctr        = vec3i();
    f.min        = blockDim;
    f.max        = vec3i();
    for (int surface = 0; surface < 6; ++surface) {
        f.boundaryCtr[surface] = vec3i();
        f.boundaryMin[surface] = vec3i();
        f.boundaryMax[surface] = vec3i();
        f.numVoxelOnSurface[surface] = 0;
    }
    f.edgeVoxels.clear();
    f.bodyVoxels.clear();
    f.touchedSurfaces.clear();
    return f;
}

void FeatureTracker::FindNewFeature(vec3i seed) {
    maskValue++;

    Feature f = createNewFeature();
    f.maskValue = maskValue;
    f.edgeVoxels.push_back(seed);

    expandRegion(f);

    if (static_cast<int>(f.bodyVoxels.size()) < MIN_NUM_VOXEL_IN_FEATURE) {
        maskValue--; 
        return;
    }

    currentFeatures.push_back(f);
    backup1Features = currentFeatures;
    backup2Features = currentFeatures;
    backup3Features = currentFeatures;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    if (tfMap.size() == 0 || tfRes <= 0) {
        std::cout << "Set TF pointer first." << std::endl;
        exit(EXIT_FAILURE);
    }

    data.assign(pData, pData+volumeSize);

    // backup mask to maskPrev then clear mask
    maskPrev.clear();
    maskPrev.swap(mask);

    for (auto i = 0; i < currentFeatures.size(); ++i) {
        Feature f = currentFeatures[i];

        vec3i offset = predictRegion(i, direction, mode);
        fillRegion(f, offset);
        shrinkRegion(f);
        expandRegion(f);

        if (static_cast<int>(f.bodyVoxels.size()) < MIN_NUM_VOXEL_IN_FEATURE) {
            // erase feature from list when it becomes too small
            currentFeatures.erase(currentFeatures.begin() + i);
            continue;
        } else {
            currentFeatures[i] = f;    
        }
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline vec3i FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timeLeft2Backward : timeLeft2Forward;

    vec3i offset;
    Feature b1f = backup1Features[index];
    Feature b2f = backup2Features[index];
    Feature b3f = backup3Features[index];

    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    offset = b2f.ctr - b1f.ctr;
                } else {  // Tracking forward as default
                    offset = b3f.ctr - b2f.ctr;
                }
                for (auto& voxel : b3f.edgeVoxels) {
                    voxel += offset;
                    voxel = util::min(voxel, blockDim-vec3i(1,1,1));   // x, y, z at most dim-1
                    voxel = util::max(voxel, vec3i());  // x, y, z at least 0
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    offset = b3f.ctr*2 - b2f.ctr*3 + b1f.ctr;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        offset = b2f.ctr - b1f.ctr;
                    } else {  // Tracking forward as default
                        offset = b3f.ctr - b2f.ctr;
                    }
                }
                for (auto& voxel : b3f.edgeVoxels) {
                    voxel += offset;
                    voxel = util::min(voxel, blockDim-vec3i(1,1,1));   // x, y, z at most dim-1
                    voxel = util::max(voxel, vec3i());  // x, y, z at least 0
                }
            }
        break;
    }
    return offset;
}

inline void FeatureTracker::fillRegion(Feature &f, const vec3i& offset) {
    // predicted to be on edge
    for (const auto& voxel : f.edgeVoxels) {
        int index = GetVoxelIndex(voxel);
        if (mask[index] == 0.0) {
            mask[index] = static_cast<float>(f.maskValue);
        }
        f.bodyVoxels.push_back(voxel);
        f.ctr += voxel;
    }

    // currently not on edge but previously on edge
    for (const auto& voxel : f.edgeVoxels) {
        vec3i voxelPrev = voxel - offset;
        int index = GetVoxelIndex(voxel);
        int indexPrev = GetVoxelIndex(voxelPrev);
        if (voxel.x >= 0 && voxel.x <= blockDim.x && voxelPrev.x >= 0 && voxelPrev.x <= blockDim.x &&
            voxel.y >= 0 && voxel.y <= blockDim.y && voxelPrev.y >= 0 && voxelPrev.y <= blockDim.y &&
            voxel.z >= 0 && voxel.z <= blockDim.z && voxelPrev.z >= 0 && voxelPrev.z <= blockDim.z &&
            mask[index] == 0.0 && maskPrev[indexPrev] == static_cast<float>(f.maskValue)) {

            // mark voxels that: 1. currently = 1; or 2. currently = 0 but previously = 1;
            mask[index] = static_cast<float>(f.maskValue);
            f.bodyVoxels.push_back(voxel);
            f.ctr += voxel;
        }
    }
}

inline void FeatureTracker::shrinkRegion(Feature &f) {
    // mark all edge points as 0
    while (!f.edgeVoxels.empty()) {
        vec3i voxel = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        shrinkEdge(f, voxel);
    }

    while (!f.bodyVoxels.empty()) {
        vec3i voxel = f.bodyVoxels.front();
        f.bodyVoxels.pop_front();

        int index = GetVoxelIndex(voxel);
        bool voxelOnEdge = false;
        if (getOpacity(data[index]) < OPACITY_THRESHOLD) {
            voxelOnEdge = false;
            // if point is invisible, mark its adjacent points as 0
            shrinkEdge(f, voxel);                                               // center
            if (++voxel.x < blockDim.x) { shrinkEdge(f, voxel); } voxel.x--;   // right
            if (++voxel.y < blockDim.y) { shrinkEdge(f, voxel); } voxel.y--;   // top
            if (++voxel.z < blockDim.z) { shrinkEdge(f, voxel); } voxel.z--;   // back
            if (--voxel.x >= 0)          { shrinkEdge(f, voxel); } voxel.x++;   // left
            if (--voxel.y >= 0)          { shrinkEdge(f, voxel); } voxel.y++;   // bottom
            if (--voxel.z >= 0)          { shrinkEdge(f, voxel); } voxel.z++;   // front
        } else if (mask[index] == 0.0f) { voxelOnEdge = true; }

        if (voxelOnEdge) { 
            f.edgeVoxels.push_back(voxel); 
        }
    }

    for (const auto& voxel : f.edgeVoxels) {
        int index = GetVoxelIndex(voxel);
        if (mask[index] != static_cast<float>(f.maskValue)) {
            mask[index] = static_cast<float>(f.maskValue);
            f.bodyVoxels.push_back(voxel);
            f.ctr += voxel;
        }
    }
}

inline void FeatureTracker::shrinkEdge(Feature& f, const vec3i& voxel) {
    int index = GetVoxelIndex(voxel);
    if (mask[index] == static_cast<float>(f.maskValue)) {
        mask[index] = 0.0;  // shrink
        auto it = std::find(f.bodyVoxels.begin(), f.bodyVoxels.end(), voxel);
        if (it != f.bodyVoxels.end()) {
            f.bodyVoxels.erase(it);    
            f.edgeVoxels.push_back(voxel);
            f.ctr -= voxel;
        }
    }
}

inline void FeatureTracker::expandRegion(Feature& f) {
    std::list<vec3i> tempVoxels;    // to store updated edge voxels
    while (!f.edgeVoxels.empty()) {
        vec3i voxel = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        bool voxelOnEdge = false;
        if (++voxel.x < blockDim.x) { voxelOnEdge |= expandEdge(f, voxel); } voxel.x--;  // right
        if (++voxel.y < blockDim.y) { voxelOnEdge |= expandEdge(f, voxel); } voxel.y--;  // top
        if (++voxel.z < blockDim.z) { voxelOnEdge |= expandEdge(f, voxel); } voxel.z--;  // front
        if (--voxel.x >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.x++;  // left
        if (--voxel.y >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.y++;  // bottom
        if (--voxel.z >= 0) { voxelOnEdge |= expandEdge(f, voxel); } voxel.z++;  // back

        // if any one of the six neighboring points is not on edge, the original
        // seed voxel is still considered as on edge and will be put back to edge list
        if (voxelOnEdge) { 
            tempVoxels.push_back(voxel); 
        }
    }
    f.edgeVoxels.swap(tempVoxels);

    // update feature info - accumulative
    if (f.bodyVoxels.size() != 0) {
        f.ctr /= f.bodyVoxels.size();
        f.id = GetVoxelIndex(f.ctr);
    }

    for (int surface = 0; surface < 6; ++surface) {
        if (f.numVoxelOnSurface[surface] != 0) {
            f.boundaryCtr[surface] /= f.numVoxelOnSurface[surface];
            if (std::find(f.touchedSurfaces.begin(), f.touchedSurfaces.end(), surface) == 
                f.touchedSurfaces.end()) {
                f.touchedSurfaces.push_back(surface);
            }           
        }
    }
}

inline bool FeatureTracker::expandEdge(Feature& f, const vec3i& voxel) {
    int index = GetVoxelIndex(voxel);

    if (mask[index] > 0 || getOpacity(data[index]) < OPACITY_THRESHOLD) {
        // this neighbor voxel is already labeled, or the opacity is not large enough
        // to be labeled as within the feature, so the original seed is still on edge.
        return true;
    }

    // update feature info
    mask[index] = static_cast<float>(f.maskValue);
    f.min = util::min(f.min, voxel);
    f.max = util::max(f.max, voxel);
    f.ctr += voxel;  // averaged later
    if (voxel.x == 0) { updateFeatureBoundary(f, voxel, LEFT);   }
    if (voxel.y == 0) { updateFeatureBoundary(f, voxel, BOTTOM); }
    if (voxel.z == 0) { updateFeatureBoundary(f, voxel, FRONT);  }
    if (voxel.x == blockDim.x-1) { updateFeatureBoundary(f, voxel, RIGHT); }
    if (voxel.y == blockDim.y-1) { updateFeatureBoundary(f, voxel, TOP);   }
    if (voxel.z == blockDim.z-1) { updateFeatureBoundary(f, voxel, BACK);  }
    f.edgeVoxels.push_back(voxel);
    f.bodyVoxels.push_back(voxel);
    // f.touchedSurfaces is updated after the expandRegion() is done

    // the original seed voxel is no longer on edge for this neighboring direction
    return false;
}

void FeatureTracker::updateFeatureBoundary(Feature& f, const vec3i& voxel, int surface) {
    f.boundaryMin[surface] = util::min(f.boundaryMin[surface], voxel);
    f.boundaryMax[surface] = util::max(f.boundaryMax[surface], voxel);
    f.boundaryCtr[surface] += voxel;
    f.numVoxelOnSurface[surface]++;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1Features = backup2Features;
    backup2Features = backup3Features;
    backup3Features = currentFeatures;

    if (direction == FT_FORWARD) {
        if (timeLeft2Forward  < 3) ++timeLeft2Forward;
        if (timeLeft2Backward > 0) --timeLeft2Backward;
    } else {    // direction is either FORWARD or BACKWARD
        if (timeLeft2Forward  > 0) --timeLeft2Forward;
        if (timeLeft2Backward < 3) ++timeLeft2Backward;
    }
}



int main (int argc, char **argv) {
if (argc != 5) {
    std::cout << "argv[0]: " << argv[0] << std::endl;
    std::cout << "argv[1]: " << argv[1] << std::endl;
    std::cout << "argv[2]: " << argv[2] << std::endl;
    std::cout << "argv[3]: " << argv[3] << std::endl;
    std::cout << "argv[4]: " << argv[4] << std::endl;
    std::cout << "Usage : " << argv[0] << " npx npy npz" << "argv[4]" << std::endl;
    return EXIT_FAILURE;
}

MpiController mc;
mc.InitWith(argc, argv);
mc.Start();

return EXIT_SUCCESS;
}

