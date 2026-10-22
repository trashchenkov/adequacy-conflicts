#include <algorithm>
#include <cmath>
#include <vector>

static int longest(int mask,int T,bool cyclic){
    if(mask==((1<<T)-1)) return T;
    int best=0,cur=0;
    for(int t=0;t<(cyclic?2*T:T);++t){
        cur=(mask&(1<<(t%T)))?cur+1:0; best=std::max(best,cur);
    }
    return best;
}

static bool feasible(const double *x,int T,int cover,double E,double Pc,double r){
    double total=0,hi=0,lo=0,hicur=0,locur=0;
    for(int t=0;t<T;++t){
        double q=(cover&(1<<t))?std::max(-x[t]-r,0.):0.;
        if(q>1.+1e-12) return false;
        double w=q-std::min(Pc,std::max(x[t]+r,0.));
        total+=w; hicur=std::max(0.,hicur+w); locur=std::min(0.,locur+w);
        hi=std::max(hi,hicur); lo=std::min(lo,locur);
    }
    return total<=1e-12 && std::max(hi,total-lo)<=E+1e-12;
}

extern "C" void evaluate(const double *x,int T,double E,double Pc,double *answer,double *table){
    const int K=T+1;
    double grid[2][13][13];
    for(int j=0;j<2;++j) for(int h=0;h<K;++h) for(int d=0;d<K;++d) grid[j][h][d]=1e30;
    int neg=0;
    for(int t=0;t<T;++t) if(x[t]<0) neg|=1<<t;
    for(int cover=neg;;cover=(cover-1)&neg){
        double lower=0,upper=0;
        for(int t=0;t<T;++t) if(cover&(1<<t)){
            lower=std::max(lower,-x[t]-1.); upper=std::max(upper,-x[t]);
        }
        if(feasible(x,T,cover,E,Pc,lower)) upper=lower;
        else for(int iter=0;iter<27;++iter){
            double mid=(lower+upper)/2;
            if(feasible(x,T,cover,E,Pc,mid)) upper=mid; else lower=mid;
        }
        if(table) table[cover]=upper;
        int shortmask=neg^cover,h=__builtin_popcount((unsigned)shortmask);
        for(int j=0;j<2;++j){int d=longest(shortmask,T,j); grid[j][h][d]=std::min(grid[j][h][d],upper);}
        if(cover==0) break;
    }
    for(int j=0;j<2;++j){
        for(int h=0;h<K;++h) for(int d=0;d<K;++d){
            if(h)grid[j][h][d]=std::min(grid[j][h][d],grid[j][h-1][d]);
            if(d)grid[j][h][d]=std::min(grid[j][h][d],grid[j][h][d-1]);
        }
        double best=-1; int bh=0,bd=0;
        for(int h=0;h<K;++h) for(int d=0;d<K;++d){
            double gamma=grid[j][h][d]-std::max(grid[j][h][T],grid[j][T][d]);
            if(gamma>best+1e-10){best=gamma;bh=h;bd=d;}
        }
        answer[7*j]=best; answer[7*j+1]=bh; answer[7*j+2]=bd;
        answer[7*j+3]=grid[j][bh][T]; answer[7*j+4]=grid[j][T][bd];
        answer[7*j+5]=grid[j][bh][bd]; answer[7*j+6]=neg;
    }
}
