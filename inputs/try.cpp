#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <map>
#include <set>
#include <string>

void ReadContentFile(   std::string& content_file, 
                        std::vector <std::vector < std::vector <double>>>& feature, 
                        std::vector <std::string>& labels, uint& layers_plus )
{
    // 读取文件
    std::vector <std::vector <std::string>> info;
    std::ifstream infile;
	infile.open(content_file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
	std::vector<std::string> data;
	std::string s;
	while (std::getline(infile, s)) {
		std::istringstream is(s); //将读出的一行转成数据流进行操作
		std::string d;
		while (!is.eof()) {
			is >> d;
			data.push_back(d);
		}
		info.push_back(data);
		data.clear();
		s.clear();
	}
	infile.close();             //关闭文件输入流 

    // 提取元素
    for (uint i = 0; i < info.size(); i++ ){
        labels[i] = info[i][info[0].size()-1];
        feature[i].resize(layers_plus);
        feature[i][0].resize(info[0].size()-2);
        for(uint j = 1; j < info[0].size()-1; j++ ){
            std::istringstream iss(info[i][j]);  
            double num;  
            iss >> num;  
            feature[i][0][j-1]= num;
            
        }
        
    }


}
void OneHot( std::vector <std::string>& labels, std::vector <std::vector <double>>& y)
{
    //one_hot 编码
    std::map <std::string,uint> mp;
    std::set <std::string> classes; // classes string类型 看有多少类
    for (auto label : labels){
        classes.insert(label);
    }
    uint num = 0;
    for(std::set<std::string>::iterator it=classes.begin() ;it!=classes.end();it++)
    {
        mp.insert(std::make_pair(*it,num++));
    }
    std::map <std::string,uint>::iterator iter;
    iter = mp.begin();
    while(iter != mp.end()) {
        std::cout << iter->first << " : " << iter->second << std::endl;
        iter++;
    }

    for( uint src = 0; src < labels.size(); src++) {
       y[src].resize(mp.size(),0);
       std::string y_true = labels[src];
       y[src][mp[y_true]] = 1;
    }

    // for (auto a : y) {
	// 	for (auto  yy: a) {
	// 	std::cout << yy << " ";
    //     }
		
	// 	std::cout << std::endl;
	// }
}

void Normalize( std::vector <double>& v)
{
    //归一化
    double sum = 0.0000;
    for (uint i = 0; i < v.size(); i++) {
    sum += v[i];
    }
    for (uint i = 0; i < v.size(); i++) {
    v[i] = v[i]/sum;
    }

}
int main(){
    
    std::vector <std::vector < std::vector <double>>> feature; // Hi
    std::vector <std::string> labels; // labels string类型
    std::vector <std::vector <double>> y; 
    
    feature.resize(2708);
    labels.resize(2708);
    y .resize(2708);
    // for (uint src =0 ; src < 10; src++){
    //     feature[src].resize(3);
    // }

    // std::vector <std::vector <std::string>> info;
    std::string file = "cora_content.txt";
    uint layers_plus = 3;
    ReadContentFile(file, feature, labels, layers_plus);
    OneHot(labels, y);
    for (uint src =0 ; src < 2708; src++){
            Normalize(feature[src][0]);
    }
        
    std::cout << feature[0][0].size()<<std::endl;
    
	// featrue 需要归一化

	// for (auto a : labels) {
		
	// 	std::cout << a << " ";
		
	// 	std::cout << std::endl;
	// }
    // for (uint i = 0; i < feature.size(); i++ ){
    //     for(uint j = 0; j < feature[0][0].size(); j++ ){
    //         std::cout << feature[i][0][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    for (uint i = 0; i < y.size(); i++ ){
        std::cout << i<<":::";
        for(uint j = 0; j < y[0].size(); j++ ){
            std::cout << y[i][j] << " ";
        }
        std::cout << std::endl;
    }
	
    return 0;
}