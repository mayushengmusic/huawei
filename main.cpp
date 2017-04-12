#include <iostream>
#include <vector>
#include <cstring>
#include <queue>
#include <list>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <map>
#include <random>
#include <stack>
#include <chrono>

#define EPSINON 0.0000001

struct NetLink{
    unsigned int bandwidth;
    unsigned int unit_price;
};



struct Node{
    float Pheromone;
    unsigned long  OutFlow;
    std::list<unsigned int> Friends;
};





bool cmpfor_u_int_big(unsigned int & a, unsigned int & b)
{
    return a>b;
}

bool cmpfor_u_int_small(unsigned int & a, unsigned int & b)
{
    return a<b;
}

bool cmpfor_pair_u_int_float_big(std::pair<unsigned int,float> & a,std::pair<unsigned int,float> &b)
{
    return a.second>b.second;
}

bool cmpfor_pair_u_int_float_small(std::pair<unsigned int,float> & a,std::pair<unsigned int,float> &b)
{
    return a.second<b.second;
}

bool cmpfor_pair_u_int_u_long_big(std::pair<unsigned int, unsigned long> &a,std::pair<unsigned int, unsigned long> &b)
{
    return a.second>b.second;
}

bool cmpfor_pair_u_int_u_long_small(std::pair<unsigned int,float> & a,std::pair<unsigned int,float> &b)
{
    return a.second<b.second;
}

class Matrix{
public:
    Matrix(unsigned int MatrixSize);
    Matrix(Matrix & src);
    void SetNetLink(unsigned int startNode,unsigned int endNode,unsigned int bandwidth, unsigned int unit_price);
    void DecreateBandwidth(unsigned int startNode, unsigned int endNode,unsigned int bandwidthfordec);
    void deleteLink(unsigned int startNode, unsigned int endNode);
    bool CheckNetLink(unsigned int startNode, unsigned int endNode);
    const NetLink & GetNetLink(unsigned int startNode, unsigned int endNode);
    const std::list<unsigned int> & GetFriendNodes(const unsigned int &Node);
    const float & GetPheromone(const unsigned int &Node);
    void SetPheromone(const unsigned int &Node, const float & new_Pheromone);
    void PheromoneIncreate(const unsigned int & Node,float update);
    const long int  GetOutFlow(const unsigned int &Node);
    void PheromoneDecreate(float rate);
    unsigned int GetMatrixSize();
    void SetSuperStart(std::list<unsigned int> & listcontainerstart);
    void SetSuperEnd(unsigned int end, unsigned int requestflow);
    void show();



    ~Matrix();

private:
    struct NetLink *LinkData;
    struct Node * NodeData;
    unsigned int CurrentSize;


};

Matrix::Matrix(unsigned int MatrixSize) : CurrentSize(MatrixSize+2){
    LinkData = new struct NetLink[CurrentSize*CurrentSize];
    for(unsigned int i=0;i<CurrentSize*CurrentSize;i++)
    {
        LinkData[i].unit_price=0;
        LinkData[i].bandwidth=0;
    }

    NodeData = new struct Node[CurrentSize];
    for(unsigned int i=0;i<CurrentSize;i++) {
        NodeData[i].OutFlow = 0;
        NodeData[i].Pheromone=100.0;
        NodeData[i].Friends=std::list<unsigned int>();
    }
}

Matrix::Matrix(Matrix &src):CurrentSize(src.GetMatrixSize()) {
    LinkData = new struct NetLink[CurrentSize*CurrentSize];
    for(unsigned int i=0;i<CurrentSize*CurrentSize;i++)
    {
        LinkData[i].unit_price=src.LinkData[i].unit_price;
        LinkData[i].bandwidth=src.LinkData[i].bandwidth;
    }

    NodeData = new struct Node[CurrentSize];
    for(unsigned int i=0;i<CurrentSize;i++) {
        NodeData[i].OutFlow = src.NodeData[i].OutFlow;
        NodeData[i].Pheromone=src.NodeData[i].Pheromone;
        NodeData[i].Friends=src.NodeData[i].Friends;
    }
}

Matrix::~Matrix() {
    delete[] LinkData;
    delete[] NodeData;
}


void Matrix::deleteLink(unsigned int startNode, unsigned int endNode) {
    if(CheckNetLink(startNode,endNode)) {
        NodeData[startNode].OutFlow -= (((LinkData + startNode * CurrentSize) + endNode)->bandwidth);
        ((LinkData + startNode * CurrentSize) + endNode)->bandwidth = 0;
        ((LinkData + startNode * CurrentSize) + endNode)->unit_price = 0;

        NodeData[startNode].Friends.remove(endNode);
    }

}


void Matrix::DecreateBandwidth(unsigned int startNode, unsigned int endNode, unsigned int bandwidthfordec) {

        NodeData[startNode].OutFlow-=bandwidthfordec;
        ((LinkData+startNode*CurrentSize)+endNode)->bandwidth-=bandwidthfordec;
        if(((LinkData+startNode*CurrentSize)+endNode)->bandwidth==0)
        {
            ((LinkData + startNode * CurrentSize) + endNode)->unit_price = 0;
            NodeData[startNode].Friends.remove(endNode);
        }

}

void Matrix::SetNetLink(unsigned int startNode, unsigned int endNode, unsigned int bandwidth,
                        unsigned int unit_price) {


    if(!CheckNetLink(startNode,endNode))
    {
        ((LinkData+startNode*CurrentSize)+endNode)->bandwidth=bandwidth;
        ((LinkData+startNode*CurrentSize)+endNode)->unit_price=unit_price;

        ((LinkData+endNode*CurrentSize)+startNode)->bandwidth=bandwidth;
        ((LinkData+endNode*CurrentSize)+startNode)->unit_price=unit_price;

        NodeData[startNode].Friends.push_back(endNode);
        NodeData[endNode].Friends.push_back(startNode);

        NodeData[startNode].OutFlow+=bandwidth;
        NodeData[endNode].OutFlow+=bandwidth;
    }




}


void Matrix::show() {
    for(unsigned int i=0;i<CurrentSize;i++)
    {
        for(unsigned int j=0;j<CurrentSize;j++)
            std::cout<<"("<<(LinkData+i*CurrentSize+j)->bandwidth<<","<<(LinkData+i*CurrentSize+j)->unit_price<<") ";
        std::cout<<std::endl;
    }

}

const NetLink & Matrix::GetNetLink(unsigned int startNode, unsigned int endNode) {
    return *((LinkData+startNode*CurrentSize)+endNode);
}

bool Matrix::CheckNetLink(unsigned int startNode, unsigned int endNode) {
    return (((LinkData+startNode*CurrentSize)+endNode)->bandwidth)&&(((LinkData+startNode*CurrentSize)+endNode)->unit_price);
}

const std::list<unsigned int> & Matrix::GetFriendNodes(const unsigned int &Node) {
    return NodeData[Node].Friends;
}

const float& Matrix::GetPheromone(const unsigned int &Node) {
    return NodeData[Node].Pheromone;
}

const long int Matrix::GetOutFlow(const unsigned int &Node) {
    return NodeData[Node].OutFlow;
}

void Matrix::SetPheromone(const unsigned int &Node,const float & new_Pheromone) {
    NodeData[Node].Pheromone=new_Pheromone;
}

void Matrix::PheromoneIncreate(const unsigned int &Node, float update) {
    NodeData[Node].Pheromone+=update;
}

void Matrix::PheromoneDecreate(float rate) {
    for(unsigned int i=0;i<CurrentSize;i++)
    {
            NodeData[i].Pheromone*=rate;
    }
}

unsigned int Matrix::GetMatrixSize() {
    return CurrentSize;
}

void Matrix::SetSuperStart(std::list<unsigned int> &listcontainerstart) {
    static unsigned int supperStart = CurrentSize-2;

    if(!NodeData[supperStart].Friends.empty())
    {
        for(auto &x: NodeData[supperStart].Friends)
        {
            deleteLink(supperStart,x);

        }
    }

    NodeData[supperStart].Friends.clear();

    for(auto x: listcontainerstart)
    {
        if(!CheckNetLink(supperStart,x)) {
            ((LinkData + supperStart * CurrentSize) + x)->bandwidth = (unsigned int) GetOutFlow(x);
            ((LinkData + supperStart * CurrentSize) + x)->unit_price = 0;

            NodeData[supperStart].Friends.push_back(x);
        }
    }

}

void Matrix::SetSuperEnd(unsigned int end, unsigned int requestflow) {
    static unsigned int supperEnd = CurrentSize-1;
    ((LinkData+end*CurrentSize)+supperEnd)->bandwidth=requestflow;
    ((LinkData+end*CurrentSize)+supperEnd)->unit_price=0;

    NodeData[end].Friends.push_back(supperEnd);
    NodeData[end].OutFlow+=requestflow;
}



/*

*/


struct cmp{
    bool operator()(std::pair<unsigned int,unsigned long *> & a,std::pair<unsigned int,unsigned long*> & b){
        return *(a.second)>*(b.second);
    }
};


class FindMaxFlow{
public:
    FindMaxFlow(Matrix* matrix);
    ~FindMaxFlow();
    unsigned long Dijkstra(const unsigned int &start, const unsigned int &end);
    std::pair<unsigned long, unsigned long> CalcuateMaxFlow(const unsigned int &start, const unsigned int &end);
    void FindPath(std::list<std::vector<unsigned int>> &pathandpathflow, unsigned int start, unsigned int end);
    bool checkwork(unsigned int point)
    {
        return *(checkworked+point);
    }

    unsigned long GetNodeFlow(unsigned int point)
    {
        return *(nodeflow+point);
    }

private:
    unsigned int FindLimitpathAndCut(const unsigned int &start,const unsigned int &end);
    Matrix matrix;
    unsigned long *dist;
    bool *visable;
    unsigned int *pathdic;
    bool *checkworked;
    unsigned long *nodeflow;

};

FindMaxFlow::FindMaxFlow(Matrix *matrix):matrix(*matrix) {
    dist=new unsigned long[matrix->GetMatrixSize()];
    pathdic= new unsigned [matrix->GetMatrixSize()];
    visable=new bool[matrix->GetMatrixSize()];
    checkworked=new bool[matrix->GetMatrixSize()];
    nodeflow=new unsigned long[matrix->GetMatrixSize()];
}


FindMaxFlow::~FindMaxFlow() {
    delete[] dist;
    delete[] visable;
    delete[] pathdic;
    delete[] checkworked;
    delete[] nodeflow;
}


void FindMaxFlow::FindPath(std::list<std::vector<unsigned int>> &pathandpathflow, unsigned int start, unsigned int end) {

    unsigned long shortpathdis=Dijkstra(start,end);
    while(shortpathdis!=std::numeric_limits<unsigned long>::max())
    {
        pathandpathflow.push_back(std::vector<unsigned int>());

        unsigned int direct = end;
        pathandpathflow.back().push_back(direct);

        while(direct!=start)
        {
            direct=pathdic[direct];
            pathandpathflow.back().push_back(direct);
        }

        unsigned int limitflow = FindLimitpathAndCut(start,end);
        pathandpathflow.back().push_back(limitflow);
        shortpathdis=Dijkstra(start,end);

    }

}



unsigned long FindMaxFlow::Dijkstra(const unsigned int &start, const unsigned int &end) {

    std::priority_queue<std::pair<unsigned int,unsigned long*>,std::vector<std::pair<unsigned int,unsigned long*>>,cmp> Q;

    for(unsigned int i=0;i<matrix.GetMatrixSize();i++)
        dist[i]=std::numeric_limits<unsigned long>::max();

    memset(visable,0,sizeof(bool)*matrix.GetMatrixSize());
    memset(pathdic,0,sizeof(unsigned int)*matrix.GetMatrixSize());

    dist[start]=0;
    Q.push(std::make_pair(start,dist+start));



    while(!Q.empty()) {

        unsigned int u=Q.top().first;
        Q.pop();
        visable[u]=true;
        if(u==end)
            break;

        for (auto &x: matrix.GetFriendNodes(u)) {


            if(dist[x]>(dist[u]+matrix.GetNetLink(u, x).unit_price))
            {
                dist[x]=dist[u]+matrix.GetNetLink(u, x).unit_price;
                pathdic[x]=u;
            }


            if(!visable[x]) {
                Q.push(std::make_pair(x,dist+x));
            }
        }

    }


    return dist[end];

}

unsigned int FindMaxFlow::FindLimitpathAndCut(const unsigned int &start, const unsigned int &end) {

   //static unsigned int min_path_start;
   //static unsigned int min_path_end;
    unsigned int limitbandwidth=std::numeric_limits<unsigned int>::max();
    unsigned int direct = end;
    while(direct!=start)
    {
        unsigned int tempfront=pathdic[direct];
        if(matrix.GetNetLink(tempfront,direct).bandwidth<limitbandwidth)
        {
            limitbandwidth=matrix.GetNetLink(tempfront,direct).bandwidth;
          // min_path_end=direct;
          // min_path_start=tempfront;
        }
        direct=tempfront;
    }

  // matrix.deleteLink(min_path_start,min_path_end);

    direct=end;

    while(direct!=start)
    {
        unsigned int tempfront=pathdic[direct];
        if(tempfront==start) {
            checkworked[direct] = true;
            nodeflow[direct] +=limitbandwidth;
        }
        matrix.DecreateBandwidth(tempfront,direct,limitbandwidth);
        direct=tempfront;
    }

    return limitbandwidth;



}


std::pair<unsigned long, unsigned long> FindMaxFlow::CalcuateMaxFlow(const unsigned int &start, const unsigned int &end) {

    memset(checkworked,false,matrix.GetMatrixSize()*sizeof(bool));
    memset(nodeflow,0,matrix.GetMatrixSize()*sizeof(unsigned long));

    unsigned long maxFlow=0;
    unsigned long shortpathdis=Dijkstra(start,end);
    unsigned long minPrice=0;

    while(shortpathdis!=std::numeric_limits<unsigned long>::max())
    {
        unsigned int limitflow = FindLimitpathAndCut(start,end);
        maxFlow+=limitflow;
        minPrice+=limitflow*shortpathdis;
        shortpathdis=Dijkstra(start,end);

    }

    return std::make_pair(maxFlow,minPrice);
}


class Ant{
public:

    Ant(Matrix *map, unsigned int choosenodenum, unsigned int requestflow, unsigned int price);
    void iteration();


private:
    Matrix *sitemap;
    unsigned int chooseNodeNum;
    static float limit;
    static float decreaterate;
    static float weight_for_outflow;
    static float weight_for_pheromone;
    unsigned int requestflow;
    unsigned int serverprice;
    unsigned long basepriceinhistory;
    std::map<std::list<unsigned int>,bool> denymap;

};

float Ant::limit=0.5;
float Ant::decreaterate = 0.95;
float Ant::weight_for_outflow=1;
float Ant::weight_for_pheromone=3;

Ant::Ant(Matrix *map, unsigned int choosenodenum, unsigned int requestflow, unsigned int price):sitemap(map),
                                                                            chooseNodeNum(choosenodenum),
                                                                            requestflow(requestflow),
                                                                            serverprice(price),
                                                                            basepriceinhistory(price*choosenodenum),
                                                                            denymap()
{

}

void Ant::iteration() {

    sitemap->PheromoneDecreate(decreaterate);
    std::list<std::pair<unsigned int,unsigned long>> findmaxcontainer;////////////////
    static unsigned int sitemap_size = sitemap->GetMatrixSize();

    for(unsigned int i=0;i<sitemap_size;i++)
    {
        static unsigned long weight_temp=0;
        weight_temp=(unsigned long)(std::pow(sitemap->GetPheromone(i),weight_for_pheromone)*std::pow(sitemap->GetOutFlow(i),weight_for_outflow));
        findmaxcontainer.push_back(std::make_pair(i,weight_temp));
    }

    std::random_device randomgen;
    float random_num = (float)((randomgen()%1000)/1000.0);
    static std::list<unsigned int> listforchoosednode;
    listforchoosednode.clear();

    static unsigned int supperstart=sitemap_size-2;
    static unsigned int supperend=sitemap_size-1;


    if(random_num>limit) {
        std::vector<std::pair<unsigned int,unsigned long>> tempvec;
        tempvec.reserve(findmaxcontainer.size());
        for(auto & x: findmaxcontainer)
            tempvec.push_back(x);
        std::random_shuffle(tempvec.begin(), tempvec.end());
        findmaxcontainer.clear();
        for(auto & x: tempvec)
           findmaxcontainer.push_back(x);
    }
    else
        findmaxcontainer.sort(cmpfor_pair_u_int_u_long_big);


    auto findmaxcontainerit=findmaxcontainer.begin();
    for(unsigned int i=0;i<chooseNodeNum;i++) {
        listforchoosednode.push_back((findmaxcontainerit++)->first);
    }

    listforchoosednode.sort(cmpfor_u_int_small);

    while(denymap[listforchoosednode])
    {

        listforchoosednode.clear();
        std::vector<std::pair<unsigned int,unsigned long>> tempvec;
        tempvec.reserve(findmaxcontainer.size());
        for(auto & x: findmaxcontainer)
            tempvec.push_back(x);
        std::random_shuffle(tempvec.begin(), tempvec.end());
        findmaxcontainer.clear();
        for(auto & x: tempvec)
            findmaxcontainer.push_back(x);

        auto findmaxcontaineritx=findmaxcontainer.begin();
        for(unsigned int i=0;i<chooseNodeNum;i++)
            listforchoosednode.push_back((findmaxcontaineritx++)->first);

        listforchoosednode.sort(cmpfor_u_int_small);

    }


    sitemap->SetSuperStart(listforchoosednode);
    FindMaxFlow solver(sitemap);
    std::pair<unsigned long,unsigned long> currentANS=solver.CalcuateMaxFlow(supperstart,supperend);

    for(auto x: sitemap->GetFriendNodes(supperstart))
    {
        if(solver.checkwork(x))
            currentANS.second+=serverprice;
    }

    std::cout<<currentANS.first<<" * "<<currentANS.second<<std::endl;

    float updatebyprice = basepriceinhistory/float(currentANS.second);

    if(currentANS.first!=requestflow&&currentANS.second>basepriceinhistory) {
        denymap[listforchoosednode] = true;
        //updatebyprice=0;
        std::cout<<"punish"<<std::endl;

    }



    if(currentANS.first==requestflow)
    {
        for(auto x: sitemap->GetFriendNodes(supperstart))
        {
            if(solver.checkwork(x)) {
                sitemap->PheromoneIncreate(x, updatebyprice*solver.GetNodeFlow(x));

            }
        }
    }

}


int main() {

   std::ifstream text("/home/jaken/test/2/case0.txt");

    /*for(int i=0;i<line_num;i++)
        text<<topo[i];
*/

    unsigned int clientNum = 0;
    std::map<unsigned int, unsigned int> PointToClient;


    unsigned int requestflow = 0;
    unsigned int iterationtime = 0;
    unsigned int serverprice = 0;
    unsigned int nodenum = 0;
    unsigned int edgenum = 0;


    text >> nodenum >> edgenum >> clientNum;


    text >> serverprice;

    Matrix test(nodenum);

    for (int i = 0; i < edgenum; i++) {
        static unsigned int startnode_temp;
        static unsigned int endnode_temp;
        static unsigned int bandwidth_temp;
        static unsigned int unit_price_temp;
        text >> startnode_temp >> endnode_temp >> bandwidth_temp >> unit_price_temp;
        //std::cout << startnode_temp << " " << endnode_temp << " " << bandwidth_temp << " " << unit_price_temp << std::endl;//
        test.SetNetLink(startnode_temp, endnode_temp, bandwidth_temp, unit_price_temp);
    }


    unsigned int limitclientnum=0;
    std::list<std::pair<unsigned int,unsigned int>> limitclientpath;

    float timesforflow=1.0;

    if(nodenum>500)
    {
        timesforflow=1.2;

    }

    for (unsigned int i = 0; i < clientNum; i++) {
        static unsigned int servernode_temp;
        static unsigned int client_temp;
        static unsigned int singlerequestflow;
        text >> client_temp >> servernode_temp >> singlerequestflow;
        PointToClient[servernode_temp] = client_temp;

        if(test.GetOutFlow(servernode_temp)<=timesforflow*singlerequestflow) {
            limitclientnum++;
            limitclientpath.push_back(std::make_pair(servernode_temp,singlerequestflow));
        }
        else
        {
            test.SetSuperEnd(servernode_temp, singlerequestflow);
            requestflow += singlerequestflow;
        }

        //
       // std::cout << servernode_temp << " " << client_temp << " " << singlerequestflow << std::endl;
        //
    }


    clientNum=clientNum-limitclientnum;


/*
    Matrix test(6);
    test.SetNetLink(0,1,7,1);
    test.SetNetLink(0,2,9,1);
    test.SetNetLink(0,5,14,1);
    test.SetNetLink(1,2,10,1);
    test.SetNetLink(1,3,15,1);
    test.SetNetLink(2,3,11,1);
    test.SetNetLink(2,5,12,1);
    test.SetNetLink(3,4,6,1);
    test.SetNetLink(4,5,9,1);
*/

    /*   Matrix test(4);
       test.SetNetLink(0,1,5,1);
       test.SetNetLink(1,2,35,1);
       test.SetNetLink(2,3,5,1);
       test.SetNetLink(0,2,30,1);
       test.SetNetLink(1,3,30,5);
       test.SetPheromone(3,200);
       test.SetPheromone(2,200);

   */
    unsigned int usingnodenum = (unsigned int) std::pow(clientNum, 0.5);

    Ant iterationAnt(&test, clientNum, requestflow, serverprice);

    iterationtime=clientNum*500;

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < iterationtime; i++) {
        iterationAnt.iteration();
        //std::cout<<i<<" iteration"<<std::endl;
        auto end = std::chrono::system_clock::now();
       // std::cout<<std::chrono::nanoseconds(end-start).count()/1000000000.0<<std::endl;//
        if(std::chrono::nanoseconds(end-start).count()>=60000000000)
                break;
    }

    std::list<std::pair<unsigned int, float>> listforchoose;

    for (unsigned int i = 0; i < nodenum; i++)
        listforchoose.push_back(std::make_pair(i, test.GetPheromone(i)));



    listforchoose.sort(cmpfor_pair_u_int_float_big);




    unsigned int superstart = test.GetMatrixSize() - 2;
    unsigned int superend = test.GetMatrixSize() - 1;

    std::list<unsigned int> choosednode;
    std::list<unsigned int> bestnode;

    unsigned long bestprice=std::numeric_limits<unsigned long>::max();

    unsigned int step = usingnodenum;
    unsigned int startnum=clientNum/2;


    if(nodenum<400)
    {
        step=3;//***********************************

    }

    if(nodenum<200)
    {
        step = 1;
        startnum=1;

    }


    for (unsigned int searchtime = 4; searchtime < clientNum; searchtime+=step)
    {
        choosednode.clear();

       //std::cout<<"search: "<<searchtime<<std::endl;

        auto listforchooseit=listforchoose.begin();
    for (unsigned int i = 0; i < searchtime; i++) {
        choosednode.push_back((listforchooseit++)->first);
        std::cout<<(listforchooseit++)->second<<" * "<<std::endl;
    }

    test.SetSuperStart(choosednode);

    FindMaxFlow solver(&test);


    auto ANS = solver.CalcuateMaxFlow(superstart, superend);

    for (auto &x: test.GetFriendNodes(superstart)) {
        if (solver.checkwork(x))
            ANS.second += serverprice;
    }
      //
      //std::cout<<ANS.first<<" "<<ANS.second+limitclientnum*serverprice<<" "<<requestflow<<std::endl;
        //
    if(ANS.first==requestflow&&ANS.second<bestprice)
    {
        bestnode=choosednode;
        bestprice=ANS.second;
    }

        auto end = std::chrono::system_clock::now();
       // std::cout<<std::chrono::nanoseconds(end-start).count()/1000000000.0<<std::endl;//
        //if(std::chrono::nanoseconds(end-start).count()>=70000000000)
        //    break;

}

  // std::ofstream outans(filename);
    if(bestprice==std::numeric_limits<unsigned long>::max())
    {
        std::cout<<"NA"<<std::endl;
        return 0;
    }
    bestprice+=limitclientnum*serverprice;

  //  std::cout<<"best node num: "<<bestnode.size()+limitclientnum<<std::endl;

    std::cout<<"best price: "<<bestprice<<"/"<<clientNum*serverprice+limitclientnum*serverprice<<std::endl;

    test.SetSuperStart(bestnode);

    FindMaxFlow solverforpath(&test);
    std::list<std::vector<unsigned int>> flowpaths;
    solverforpath.FindPath(flowpaths,superstart,superend);

    std::cout<<flowpaths.size()+limitclientpath.size()<<std::endl;
    std::cout<<std::endl;
    for(auto & path: flowpaths)
    {
        for(auto it=path.end()-3;it!=path.begin();it--)
            std::cout<<*it<<" ";

        std::cout<<PointToClient[*(path.begin()+1)]<<" "<<*(path.end()-1)<<std::endl;
    }

    for(auto & path: limitclientpath)
    {
        std::cout<<path.first<<" "<<PointToClient[path.first]<<" "<<path.second<<std::endl;
    }


    return 0;
}