#ifndef UF2_H
#define UF2_H

#include <iostream>
#include <vector>
#include <memory>
#include <iterator>
#include <limits>
#include <set>
#include <map>
#include "sparse_matrix_util.hpp"
#include "bp.hpp"
#include "osd.hpp"
#include <robin_map.h>
#include <robin_set.h>


namespace uf{

const std::vector<double> NULL_DOUBLE_VECTOR = {};

std::vector<int> sort_indices(std::vector<double>& B){
    std::vector<int> indices(B.size());
    std::iota(indices.begin(),indices.end(),0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return B[i] < B[j];});
    return indices;
}


struct Cluster{
    bp::BpSparse& pcm;
    int cluster_id;
    bool active;
    bool valid;
    tsl::robin_set<int> bit_nodes;
    tsl::robin_set<int> check_nodes;
    tsl::robin_set<int> boundary_check_nodes;
    std::vector<int> candidate_bit_nodes;
    tsl::robin_set<int> enclosed_syndromes;
    tsl::robin_map<int,int> spanning_tree_check_roots;
    tsl::robin_set<int> spanning_tree_bits;
    tsl::robin_set<int> spanning_tree_leaf_nodes;

    Cluster** global_check_membership;
    Cluster** global_bit_membership;
    tsl::robin_set<Cluster*> merge_list;

    std::vector<int> cluster_decoding;
    std::vector<int> matrix_to_cluster_bit_map;
    tsl::robin_map<int,int> cluster_to_matrix_bit_map;
    std::vector<int> matrix_to_cluster_check_map;
    tsl::robin_map<int,int> cluster_to_matrix_check_map;


    Cluster(bp::BpSparse& parity_check_matrix, int syndrome_index, Cluster** ccm, Cluster** bcm):
        pcm(parity_check_matrix){
        
        this->active=true;
        this->valid=false;
        this->cluster_id = syndrome_index;
        this->boundary_check_nodes.insert(syndrome_index);
        this->check_nodes.insert(syndrome_index);
        this->enclosed_syndromes.insert(syndrome_index);
        this->global_check_membership = ccm;
        this->global_bit_membership = bcm;
        this->global_check_membership[syndrome_index]=this;



    }
    ~Cluster(){
        this->bit_nodes.clear();
        this->check_nodes.clear();
        this->boundary_check_nodes.clear();
        this->candidate_bit_nodes.clear();
        this->enclosed_syndromes.clear();
        this->merge_list.clear();
    }

    int parity(){
        return this->enclosed_syndromes.size() % 2;
    }

    void get_candidate_bit_nodes(){

        std::vector<int> erase_boundary_check;
        this->candidate_bit_nodes.clear();
        for(int check_index: boundary_check_nodes){
            bool erase = true;
            for(auto& e: this->pcm.iterate_row(check_index)){
                if(this->global_bit_membership[e.col_index] != this ){
                    candidate_bit_nodes.push_back(e.col_index);
                    erase = false;
                }
            }
            if(erase) erase_boundary_check.push_back(check_index);
        }

        for(int check_index: erase_boundary_check){
            this->boundary_check_nodes.erase(check_index);
        }
    
    }

    int add_bit_node_to_cluster(int bit_index){

        // cout<<"Add bit node function. Bit: "<<bit_index<<endl;

        auto bit_membership = this->global_bit_membership[bit_index];
        if(bit_membership == this) return 0; //if the bit is already in the cluster terminate.
        else if(bit_membership == NULL){
            //if the bit has not yet been assigned to a cluster we add it.
            this->bit_nodes.insert(bit_index);
            this->global_bit_membership[bit_index] = this;
        }
        else{
            //if the bit already exists in a cluster, we mark down that this cluster should be
            //merged with the exisiting cluster.
            this->merge_list.insert(bit_membership);
            this->global_bit_membership[bit_index] = this;
        }

        for(auto& e: this->pcm.iterate_column(bit_index)){
            int check_index = e.row_index;
            auto check_membership = this->global_check_membership[check_index]; 
            if(check_membership == this) continue;
            else if (check_membership == NULL){
                this->check_nodes.insert(check_index);
                this->boundary_check_nodes.insert(check_index);
                this->global_check_membership[check_index] = this;
            }
            else{
                this->check_nodes.insert(check_index);
                this->boundary_check_nodes.insert(check_index);
                this->merge_list.insert(check_membership);
                this->global_check_membership[check_index] = this;
            }
        }

        return 1;

    }

    void merge_with_cluster(Cluster* cl2){

        // cout<<"Hello from merge function"<<endl;

        for(auto bit_index: cl2->bit_nodes){
            this->bit_nodes.insert(bit_index);
            this->global_bit_membership[bit_index] = this;
        }

        // cout<<"bit nodes copied"<<endl;

        for(auto check_index: cl2->check_nodes){
            this->check_nodes.insert(check_index);
            this->global_check_membership[check_index] = this;
        }

        // cout<<"check nodes copied"<<endl;

        for(auto check_index: cl2->boundary_check_nodes){
            this->boundary_check_nodes.insert(check_index);
        }

        // cout<<"boundary check nodes copied"<<endl;

        cl2->active = false;
        for(auto j: cl2->enclosed_syndromes){
            this->enclosed_syndromes.insert(j);
        }
    }

    int grow_cluster(const std::vector<double>& bit_weights = NULL_DOUBLE_VECTOR, int bits_per_step = 0){
        if(!this->active) return 0; 
        this->get_candidate_bit_nodes();

        this->merge_list.clear();

        if(bit_weights == NULL_DOUBLE_VECTOR){
            for(int bit_index: this->candidate_bit_nodes){
                this->add_bit_node_to_cluster(bit_index);
            }
        }

        else{
            std::vector<double> cluster_bit_weights;
            for(int bit: this->candidate_bit_nodes){
                cluster_bit_weights.push_back(bit_weights[bit]);
            }
            auto sorted_indices = sort_indices(cluster_bit_weights);
            int count = 0;
            for(int i: sorted_indices){
                if(count == bits_per_step) break;
                int bit_index = this->candidate_bit_nodes[i];
                this->add_bit_node_to_cluster(bit_index);
                count++;
            }

        }

        // cout<<"Before merge"<<endl;

        // for(auto cl: merge_list) cout<<cl<<" ";
        // cout<<endl;

        for(auto cl: merge_list){
            this->merge_with_cluster(cl);
            cl->active = false;
        }
        return 1;
    }

    int find_spanning_tree_parent(int check_index){
            int parent = this->spanning_tree_check_roots[check_index];
            if(parent != check_index){
                return find_spanning_tree_parent(parent);
            }
            else return parent;
        }

    void find_spanning_tree(){

        this->spanning_tree_bits.clear();
        this->spanning_tree_check_roots.clear();
        this->spanning_tree_leaf_nodes.clear();

        for(int bit_index: this->bit_nodes){
            this->spanning_tree_bits.insert(bit_index);
        }

        for(int check_index: this->check_nodes){
            this->spanning_tree_check_roots[check_index] = check_index;
        }

        int check_neighbours[2];
        for(int bit_index: this->bit_nodes){
            check_neighbours[0] = this->pcm.column_heads[bit_index]->up->row_index;
            check_neighbours[1] = this->pcm.column_heads[bit_index]->down->row_index;
        
            int root0 = this->find_spanning_tree_parent(check_neighbours[0]);
            int root1 = this->find_spanning_tree_parent(check_neighbours[1]);

            if(root0!=root1){
                this->spanning_tree_check_roots[root1] = root0;
            }
            else{
                // cout<<bit_index<<endl;
                this->spanning_tree_bits.erase(bit_index);
            }
        }

        for(int check_index: this->check_nodes){
            int spanning_tree_connectivity = 0;
            for(auto& e: this->pcm.iterate_row(check_index)){
                if(this->spanning_tree_bits.contains(e.col_index)) spanning_tree_connectivity+=1;
            }
            if(spanning_tree_connectivity == 1) this->spanning_tree_leaf_nodes.insert(check_index);
        }

    }

    std::vector<int> peel_decode(const std::vector<uint8_t>& syndrome){
        std::vector<int> erasure;
        tsl::robin_set<int> synds;
        for(auto check_index: check_nodes){
            if(syndrome[check_index] == 1) synds.insert(check_index);
        }

        this->find_spanning_tree();
        while(synds.size()>0){

            int leaf_node_index = *(this->spanning_tree_leaf_nodes.begin());
            int bit_index = -1;
            int check2 = -1;

            for(auto& e: this->pcm.iterate_row(leaf_node_index)){
                bit_index = e.col_index;
                if(this->spanning_tree_bits.contains(bit_index)) break;
            }


            for(auto& e: this->pcm.iterate_column(bit_index)){
                if(e.row_index!=leaf_node_index) check2 = e.row_index;
            }



            if(synds.contains(leaf_node_index)){
                this->spanning_tree_leaf_nodes.erase(leaf_node_index); 
                // this->spanning_tree_leaf_nodes.insert(check2);
                erasure.push_back(bit_index);
                this->spanning_tree_bits.erase(bit_index);
                if(synds.contains(check2)) synds.erase(check2);
                else synds.insert(check2);
                synds.erase(leaf_node_index);
            }
            else{
                this->spanning_tree_leaf_nodes.erase(leaf_node_index); 
                // this->spanning_tree_leaf_nodes.insert(check2);
                this->spanning_tree_bits.erase(bit_index);
            }

            //check whether new check node is a leaf
            int spanning_tree_connectivity = 0;
            for(auto& e: this->pcm.iterate_row(check2)){
                if(this->spanning_tree_bits.contains(e.col_index)) spanning_tree_connectivity+=1;
            }
            if(spanning_tree_connectivity == 1) this->spanning_tree_leaf_nodes.insert(check2);

        }

        return erasure;
    }

    // bp::BpSparse* convert_to_matrix(const std::vector<double>& bit_weights = NULL_DOUBLE_VECTOR){

    //     this->matrix_to_cluster_bit_map.clear();
    //     this->matrix_to_cluster_check_map.clear();
    //     this->cluster_to_matrix_bit_map.clear();
    //     this->cluster_to_matrix_check_map.clear();


    //     if(bit_weights!=NULL_DOUBLE_VECTOR){
    //         std::vector<double> cluster_bit_weights;
    //         std::vector<int> bit_nodes_temp;
    //         for(int bit: this->bit_nodes){
    //             cluster_bit_weights.push_back(bit_weights[bit]);
    //             bit_nodes_temp.push_back(bit);
    //         }
    //         auto sorted_indices = sort_indices(cluster_bit_weights);
    //         int count = 0;
    //         for(int i: sorted_indices){
    //             int bit_index = bit_nodes_temp[i];
    //             this->matrix_to_cluster_bit_map.push_back(bit_index);
    //             this->cluster_to_matrix_bit_map[bit_index] = count;
    //             count++;
    //         }
    //     }

    //     else{
    //         int count = 0;
    //         for(int bit_index: this->bit_nodes){
    //             this->matrix_to_cluster_bit_map.push_back(bit_index);
    //             this->cluster_to_matrix_bit_map[bit_index] = count;
    //             count++;
    //         }

    //     }
    
    //     int count = 0;

    //     for(int check_index: this->check_nodes){
    //         this->matrix_to_cluster_check_map.push_back(check_index);
    //         this->cluster_to_matrix_check_map[check_index] = count;
    //         count++;
    //     }

    //     bp::BpSparse* cluster_pcm = new bp::BpSparse(this->check_nodes.size(),this->bit_nodes.size());

    //     for(int check_index: this->check_nodes){
    //         for(auto& e: this->pcm.iterate_row(check_index)){
    //             int bit_index = e.col_index;
    //             if(this->bit_nodes.contains(bit_index)){
    //                 int matrix_bit_index = cluster_to_matrix_bit_map[bit_index];
    //                 int matrix_check_index = cluster_to_matrix_check_map[check_index];
    //                 cluster_pcm.insert_entry(matrix_check_index,matrix_bit_index,1);
    //             }
    //         }
    //     }

    //     return cluster_pcm;

    // }


    // std::vector<int> invert_decode(const std::vector<uint8_t>& syndrome, const std::vector<double>& bit_weights){
    //     auto cluster_pcm = this->convert_to_matrix(bit_weights);
    //     std::vector<uint8_t> cluster_syndrome;
    //     for(int check_index: check_nodes) cluster_syndrome.push_back(syndrome[check_index]);

    //     std::vector<uint8_t> cluster_solution;
    //     cluster_solution.resize(this->bit_nodes.size(),0);
    //     cluster_solution = cluster_pcm.lu_solve(cluster_syndrome,cluster_solution);
    //     std::vector<uint8_t> candidate_cluster_syndrome;
    //     candidate_cluster_syndrome.resize(cluster_syndrome.size());
    //     candidate_cluster_syndrome = cluster_pcm.mulvec(cluster_solution,candidate_cluster_syndrome);

    //     bool equal = true;
    //     for(int i =0; i<cluster_syndrome.size(); i++){
    //         if(cluster_syndrome[i]!=candidate_cluster_syndrome[i]){
    //             equal = false;
    //             break;
    //         }
    //     }

    //     this->cluster_decoding.clear();
    //     this->valid = equal;
    //     for(int i = 0; i<cluster_solution.size(); i++){
    //         if(cluster_solution[i] == 1) this->cluster_decoding.push_back(this->matrix_to_cluster_bit_map[i]);
    //     }

    //     delete cluster_pcm;

    //     return this->cluster_decoding;

    // }

    // std::vector<int> bposd_decode(const std::vector<uint8_t>& syndrome){
    //     auto cluster_pcm = this->convert_to_matrix();
    //     std::vector<uint8_t> cluster_syndrome;
    //     for(int check_index: check_nodes) cluster_syndrome.push_back(syndrome[check_index]);

    //     std::vector<uint8_t> cluster_solution;
    //     cluster_solution.resize(this->bit_nodes.size(),0);
        
    //     std::vector<double> bp_channel_probs;
    //     bp_channel_probs.resize(cluster_pcm.n,0.05);

    //     BPDecoder* bpd = new BPDecoder(cluster_pcm,bp_channel_probs,0,0,0,1);
    //     bposd_decoder* bposd = new bposd_decoder(bpd,0,0);
        
    //     cluster_solution = cluster_pcm.lu_solve(cluster_syndrome,cluster_solution);
        
        
    //     std::vector<uint8_t> candidate_cluster_syndrome;
    //     candidate_cluster_syndrome.resize(cluster_syndrome.size());
    //     candidate_cluster_syndrome = cluster_pcm.mulvec(cluster_solution,candidate_cluster_syndrome);

    //     bool equal = true;
    //     for(int i =0; i<cluster_syndrome.size(); i++){
    //         if(cluster_syndrome[i]!=candidate_cluster_syndrome[i]){
    //             equal = false;
    //             break;
    //         }
    //     }

    //     this->cluster_decoding.clear();
    //     this->valid = equal;
    //     for(int i = 0; i<cluster_solution.size(); i++){
    //         if(cluster_solution[i] == 1) this->cluster_decoding.push_back(this->matrix_to_cluster_bit_map[i]);
    //     }

    //     delete bposd;
    //     delete bpd;
    //     delete cluster_pcm;

    //     return this->cluster_decoding;

    // }

    void print();

};


class UfDecoder{

    private:
        bool weighted;
        bp::BpSparse& pcm;

    public:
        std::vector<uint8_t> decoding;
        int bit_count;
        int check_count;
        UfDecoder(bp::BpSparse& parity_check_matrix): pcm(parity_check_matrix){
            this->bit_count = pcm.n;
            this->check_count = pcm.m;
            this->decoding.resize(this->bit_count);
            this->weighted = false;
        }

        std::vector<uint8_t>& peel_decode(const std::vector<uint8_t>& syndrome, const std::vector<double>& bit_weights = NULL_DOUBLE_VECTOR, int bits_per_step = 1){

            fill(this->decoding.begin(), this->decoding.end(), 0);

            std::vector<Cluster*> clusters;
            std::vector<Cluster*> invalid_clusters;
            Cluster** global_bit_membership = new Cluster*[pcm.n]();
            Cluster** global_check_membership = new Cluster*[pcm.m]();

            for(int i =0; i<this->pcm.m; i++){
                if(syndrome[i] == 1){
                    Cluster* cl = new Cluster(this->pcm, i, global_check_membership, global_bit_membership);
                    clusters.push_back(cl);
                    invalid_clusters.push_back(cl);
                }
            }

            while(invalid_clusters.size()>0){

                for(auto cl: invalid_clusters){
                    if(cl->active){
                        cl->grow_cluster(bit_weights,bits_per_step);
                    }
                }

                invalid_clusters.clear();
                for(auto cl: clusters){
                    if(cl->active == true && cl->parity() == 1){
                        invalid_clusters.push_back(cl);
                    }
                }

                sort(invalid_clusters.begin(), invalid_clusters.end(), [](const Cluster* lhs, const Cluster* rhs){return lhs->bit_nodes.size() < rhs->bit_nodes.size();});

            }

            for(auto cl: clusters){
                if(cl->active){
                    auto erasure = cl->peel_decode(syndrome);
                    for(int bit: erasure) this->decoding[bit] = 1;
                }
                delete cl;
            }

            delete[] global_bit_membership;
            delete[] global_check_membership;

            return this->decoding;

        }


        // std::vector<uint8_t>& matrix_decode(const std::vector<uint8_t>& syndrome, const std::vector<double>& bit_weights = NULL_DOUBLE_VECTOR, int bits_per_step = 1){

        //     fill(this->decoding.begin(), this->decoding.end(), 0);

        //     std::vector<Cluster*> clusters;
        //     std::vector<Cluster*> invalid_clusters;
        //     Cluster** global_bit_membership = new Cluster*[pcm.n]();
        //     Cluster** global_check_membership = new Cluster*[pcm.m]();

        //     for(int i =0; i<this->pcm.m; i++){
        //         if(syndrome[i] == 1){
        //             Cluster* cl = new Cluster(this->pcm, i, global_check_membership, global_bit_membership);
        //             clusters.push_back(cl);
        //             invalid_clusters.push_back(cl);
        //         }
        //     }

        //     while(invalid_clusters.size()>0){

        //         for(auto cl: invalid_clusters){
        //             if(cl->active){
        //                 cl->grow_cluster(bit_weights,bits_per_step);
        //                 auto cluster_decoding = cl->invert_decode(syndrome,bit_weights);
        //             }
        //         }

        //         invalid_clusters.clear();
        //         for(auto cl: clusters){
        //             if(cl->active == true && cl->valid == false){
        //                 invalid_clusters.push_back(cl);
        //             }
        //         }

        //         sort(invalid_clusters.begin(), invalid_clusters.end(), [](const Cluster* lhs, const Cluster* rhs){return lhs->bit_nodes.size() < rhs->bit_nodes.size();});

        //     }

        //     for(auto cl: clusters){
        //         if(cl->active){
        //             for(int bit: cl->cluster_decoding) this->decoding[bit] = 1;
        //         }
        //         delete cl;
        //     }

        //     delete[] global_bit_membership;
        //     delete[] global_check_membership;

        //     // cout<<"hello from end of C++ function"<<endl;

        //     return this->decoding;

        // }

        // std::vector<uint8_t>& bposd_decode(const std::vector<uint8_t>& syndrome, const std::vector<double>& bit_weights = NULL_DOUBLE_VECTOR, int bits_per_step = 1){

        //     fill(this->decoding.begin(), this->decoding.end(), 0);

        //     std::vector<Cluster*> clusters;
        //     std::vector<Cluster*> invalid_clusters;
        //     Cluster** global_bit_membership = new Cluster*[pcm.n]();
        //     Cluster** global_check_membership = new Cluster*[pcm.m]();

        //     for(int i =0; i<this->pcm.m; i++){
        //         if(syndrome[i] == 1){
        //             Cluster* cl = new Cluster(this->pcm, i, global_check_membership, global_bit_membership);
        //             clusters.push_back(cl);
        //             invalid_clusters.push_back(cl);
        //         }
        //     }

        //     // cout<<"After cluster initialisation"<<endl;
        //     int count = 0;
        //     while(invalid_clusters.size()>0){

        //         for(auto cl: invalid_clusters){
        //             if(cl->active){
        //                 cl->grow_cluster();
        //                 if(count>2) auto cluster_decoding = cl->bposd_decode(syndrome);
        //             }
        //         }

        //         invalid_clusters.clear();
        //         for(auto cl: clusters){
        //             if(cl->active == true && cl->valid == false){
        //                 invalid_clusters.push_back(cl);
        //             }
        //         }

        //         count++;

        //         sort(invalid_clusters.begin(), invalid_clusters.end(), [](const Cluster* lhs, const Cluster* rhs){return lhs->bit_nodes.size() < rhs->bit_nodes.size();});

        //     }

        //     for(auto cl: clusters){
        //         if(cl->active){
        //             for(int bit: cl->cluster_decoding) this->decoding[bit] = 1;
        //         }
        //         delete cl;
        //     }

        //     delete[] global_bit_membership;
        //     delete[] global_check_membership;

        //     // cout<<"hello from end of C++ function"<<endl;

        //     return this->decoding;

        // }

};

void Cluster::print(){
        cout<<"........."<<endl;
        cout<<"Cluster ID: "<<this->cluster_id<<endl;
        cout<<"Active: "<<this->active<<endl;
        cout<<"Enclosed syndromes: ";
        for(auto i: this->enclosed_syndromes) cout<<i<<" ";
        cout<<endl;
        cout<<"Cluster bits: ";
        for(auto i: this->bit_nodes) cout<<i<<" ";
        cout<<endl;
        cout<<"Cluster checks: ";
        for(auto i: this->check_nodes) cout<<i<<" ";
        cout<<endl;
        cout<<"Candidate bits: ";
        for(auto i: this->candidate_bit_nodes) cout<<i<<" ";
        cout<<endl;
        cout<<"Boundary Checks: ";
        for(auto i: this->boundary_check_nodes) cout<<i<<" ";
        cout<<endl;
        cout<<"Spanning tree: ";
        for(auto i: this->spanning_tree_bits) cout<<i<<" ";
        cout<<endl;
        cout<<"........."<<endl;
    }


}//end namespace uf

#endif