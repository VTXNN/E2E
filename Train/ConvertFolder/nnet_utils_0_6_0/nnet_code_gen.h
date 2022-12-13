#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include <iostream>
#include "nnet_helpers.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
class FillConv1DBuffer{
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        // To be implemented in subclasses
    }
};

template<class data_T, typename CONFIG_T>
class FillConv2DBuffer{
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        // To be implemented in subclasses
    }
};

//hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_2 : public FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =          0; buffer[0][1] =    data[0]; buffer[0][2] =    data[1];
            buffer[1][0] =    data[0]; buffer[1][1] =    data[1]; buffer[1][2] =    data[2];
            buffer[2][0] =    data[1]; buffer[2][1] =    data[2]; buffer[2][2] =    data[3];
            buffer[3][0] =    data[2]; buffer[3][1] =    data[3]; buffer[3][2] =    data[4];
            buffer[4][0] =    data[3]; buffer[4][1] =    data[4]; buffer[4][2] =    data[5];
            buffer[5][0] =    data[4]; buffer[5][1] =    data[5]; buffer[5][2] =    data[6];
            buffer[6][0] =    data[5]; buffer[6][1] =    data[6]; buffer[6][2] =    data[7];
            buffer[7][0] =    data[6]; buffer[7][1] =    data[7]; buffer[7][2] =    data[8];
            buffer[8][0] =    data[7]; buffer[8][1] =    data[8]; buffer[8][2] =    data[9];
            buffer[9][0] =    data[8]; buffer[9][1] =    data[9]; buffer[9][2] =   data[10];
            buffer[10][0] =    data[9]; buffer[10][1] =   data[10]; buffer[10][2] =   data[11];
            buffer[11][0] =   data[10]; buffer[11][1] =   data[11]; buffer[11][2] =   data[12];
            buffer[12][0] =   data[11]; buffer[12][1] =   data[12]; buffer[12][2] =   data[13];
            buffer[13][0] =   data[12]; buffer[13][1] =   data[13]; buffer[13][2] =   data[14];
            buffer[14][0] =   data[13]; buffer[14][1] =   data[14]; buffer[14][2] =   data[15];
            buffer[15][0] =   data[14]; buffer[15][1] =   data[15]; buffer[15][2] =   data[16];
            buffer[16][0] =   data[15]; buffer[16][1] =   data[16]; buffer[16][2] =   data[17];
            buffer[17][0] =   data[16]; buffer[17][1] =   data[17]; buffer[17][2] =   data[18];
            buffer[18][0] =   data[17]; buffer[18][1] =   data[18]; buffer[18][2] =   data[19];
            buffer[19][0] =   data[18]; buffer[19][1] =   data[19]; buffer[19][2] =   data[20];
            buffer[20][0] =   data[19]; buffer[20][1] =   data[20]; buffer[20][2] =   data[21];
            buffer[21][0] =   data[20]; buffer[21][1] =   data[21]; buffer[21][2] =   data[22];
            buffer[22][0] =   data[21]; buffer[22][1] =   data[22]; buffer[22][2] =   data[23];
            buffer[23][0] =   data[22]; buffer[23][1] =   data[23]; buffer[23][2] =   data[24];
            buffer[24][0] =   data[23]; buffer[24][1] =   data[24]; buffer[24][2] =   data[25];
            buffer[25][0] =   data[24]; buffer[25][1] =   data[25]; buffer[25][2] =   data[26];
            buffer[26][0] =   data[25]; buffer[26][1] =   data[26]; buffer[26][2] =   data[27];
            buffer[27][0] =   data[26]; buffer[27][1] =   data[27]; buffer[27][2] =   data[28];
            buffer[28][0] =   data[27]; buffer[28][1] =   data[28]; buffer[28][2] =   data[29];
            buffer[29][0] =   data[28]; buffer[29][1] =   data[29]; buffer[29][2] =   data[30];
            buffer[30][0] =   data[29]; buffer[30][1] =   data[30]; buffer[30][2] =   data[31];
            buffer[31][0] =   data[30]; buffer[31][1] =   data[31]; buffer[31][2] =   data[32];
            buffer[32][0] =   data[31]; buffer[32][1] =   data[32]; buffer[32][2] =   data[33];
            buffer[33][0] =   data[32]; buffer[33][1] =   data[33]; buffer[33][2] =   data[34];
            buffer[34][0] =   data[33]; buffer[34][1] =   data[34]; buffer[34][2] =   data[35];
            buffer[35][0] =   data[34]; buffer[35][1] =   data[35]; buffer[35][2] =   data[36];
            buffer[36][0] =   data[35]; buffer[36][1] =   data[36]; buffer[36][2] =   data[37];
            buffer[37][0] =   data[36]; buffer[37][1] =   data[37]; buffer[37][2] =   data[38];
            buffer[38][0] =   data[37]; buffer[38][1] =   data[38]; buffer[38][2] =   data[39];
            buffer[39][0] =   data[38]; buffer[39][1] =   data[39]; buffer[39][2] =   data[40];
            buffer[40][0] =   data[39]; buffer[40][1] =   data[40]; buffer[40][2] =   data[41];
            buffer[41][0] =   data[40]; buffer[41][1] =   data[41]; buffer[41][2] =   data[42];
            buffer[42][0] =   data[41]; buffer[42][1] =   data[42]; buffer[42][2] =   data[43];
            buffer[43][0] =   data[42]; buffer[43][1] =   data[43]; buffer[43][2] =   data[44];
            buffer[44][0] =   data[43]; buffer[44][1] =   data[44]; buffer[44][2] =   data[45];
            buffer[45][0] =   data[44]; buffer[45][1] =   data[45]; buffer[45][2] =   data[46];
            buffer[46][0] =   data[45]; buffer[46][1] =   data[46]; buffer[46][2] =   data[47];
            buffer[47][0] =   data[46]; buffer[47][1] =   data[47]; buffer[47][2] =   data[48];
            buffer[48][0] =   data[47]; buffer[48][1] =   data[48]; buffer[48][2] =   data[49];
            buffer[49][0] =   data[48]; buffer[49][1] =   data[49]; buffer[49][2] =   data[50];
            buffer[50][0] =   data[49]; buffer[50][1] =   data[50]; buffer[50][2] =   data[51];
            buffer[51][0] =   data[50]; buffer[51][1] =   data[51]; buffer[51][2] =   data[52];
            buffer[52][0] =   data[51]; buffer[52][1] =   data[52]; buffer[52][2] =   data[53];
            buffer[53][0] =   data[52]; buffer[53][1] =   data[53]; buffer[53][2] =   data[54];
            buffer[54][0] =   data[53]; buffer[54][1] =   data[54]; buffer[54][2] =   data[55];
            buffer[55][0] =   data[54]; buffer[55][1] =   data[55]; buffer[55][2] =   data[56];
            buffer[56][0] =   data[55]; buffer[56][1] =   data[56]; buffer[56][2] =   data[57];
            buffer[57][0] =   data[56]; buffer[57][1] =   data[57]; buffer[57][2] =   data[58];
            buffer[58][0] =   data[57]; buffer[58][1] =   data[58]; buffer[58][2] =   data[59];
            buffer[59][0] =   data[58]; buffer[59][1] =   data[59]; buffer[59][2] =   data[60];
            buffer[60][0] =   data[59]; buffer[60][1] =   data[60]; buffer[60][2] =   data[61];
            buffer[61][0] =   data[60]; buffer[61][1] =   data[61]; buffer[61][2] =   data[62];
            buffer[62][0] =   data[61]; buffer[62][1] =   data[62]; buffer[62][2] =   data[63];
            buffer[63][0] =   data[62]; buffer[63][1] =   data[63]; buffer[63][2] =   data[64];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[63]; buffer[0][1] =   data[64]; buffer[0][2] =   data[65];
            buffer[1][0] =   data[64]; buffer[1][1] =   data[65]; buffer[1][2] =   data[66];
            buffer[2][0] =   data[65]; buffer[2][1] =   data[66]; buffer[2][2] =   data[67];
            buffer[3][0] =   data[66]; buffer[3][1] =   data[67]; buffer[3][2] =   data[68];
            buffer[4][0] =   data[67]; buffer[4][1] =   data[68]; buffer[4][2] =   data[69];
            buffer[5][0] =   data[68]; buffer[5][1] =   data[69]; buffer[5][2] =   data[70];
            buffer[6][0] =   data[69]; buffer[6][1] =   data[70]; buffer[6][2] =   data[71];
            buffer[7][0] =   data[70]; buffer[7][1] =   data[71]; buffer[7][2] =   data[72];
            buffer[8][0] =   data[71]; buffer[8][1] =   data[72]; buffer[8][2] =   data[73];
            buffer[9][0] =   data[72]; buffer[9][1] =   data[73]; buffer[9][2] =   data[74];
            buffer[10][0] =   data[73]; buffer[10][1] =   data[74]; buffer[10][2] =   data[75];
            buffer[11][0] =   data[74]; buffer[11][1] =   data[75]; buffer[11][2] =   data[76];
            buffer[12][0] =   data[75]; buffer[12][1] =   data[76]; buffer[12][2] =   data[77];
            buffer[13][0] =   data[76]; buffer[13][1] =   data[77]; buffer[13][2] =   data[78];
            buffer[14][0] =   data[77]; buffer[14][1] =   data[78]; buffer[14][2] =   data[79];
            buffer[15][0] =   data[78]; buffer[15][1] =   data[79]; buffer[15][2] =   data[80];
            buffer[16][0] =   data[79]; buffer[16][1] =   data[80]; buffer[16][2] =   data[81];
            buffer[17][0] =   data[80]; buffer[17][1] =   data[81]; buffer[17][2] =   data[82];
            buffer[18][0] =   data[81]; buffer[18][1] =   data[82]; buffer[18][2] =   data[83];
            buffer[19][0] =   data[82]; buffer[19][1] =   data[83]; buffer[19][2] =   data[84];
            buffer[20][0] =   data[83]; buffer[20][1] =   data[84]; buffer[20][2] =   data[85];
            buffer[21][0] =   data[84]; buffer[21][1] =   data[85]; buffer[21][2] =   data[86];
            buffer[22][0] =   data[85]; buffer[22][1] =   data[86]; buffer[22][2] =   data[87];
            buffer[23][0] =   data[86]; buffer[23][1] =   data[87]; buffer[23][2] =   data[88];
            buffer[24][0] =   data[87]; buffer[24][1] =   data[88]; buffer[24][2] =   data[89];
            buffer[25][0] =   data[88]; buffer[25][1] =   data[89]; buffer[25][2] =   data[90];
            buffer[26][0] =   data[89]; buffer[26][1] =   data[90]; buffer[26][2] =   data[91];
            buffer[27][0] =   data[90]; buffer[27][1] =   data[91]; buffer[27][2] =   data[92];
            buffer[28][0] =   data[91]; buffer[28][1] =   data[92]; buffer[28][2] =   data[93];
            buffer[29][0] =   data[92]; buffer[29][1] =   data[93]; buffer[29][2] =   data[94];
            buffer[30][0] =   data[93]; buffer[30][1] =   data[94]; buffer[30][2] =   data[95];
            buffer[31][0] =   data[94]; buffer[31][1] =   data[95]; buffer[31][2] =   data[96];
            buffer[32][0] =   data[95]; buffer[32][1] =   data[96]; buffer[32][2] =   data[97];
            buffer[33][0] =   data[96]; buffer[33][1] =   data[97]; buffer[33][2] =   data[98];
            buffer[34][0] =   data[97]; buffer[34][1] =   data[98]; buffer[34][2] =   data[99];
            buffer[35][0] =   data[98]; buffer[35][1] =   data[99]; buffer[35][2] =  data[100];
            buffer[36][0] =   data[99]; buffer[36][1] =  data[100]; buffer[36][2] =  data[101];
            buffer[37][0] =  data[100]; buffer[37][1] =  data[101]; buffer[37][2] =  data[102];
            buffer[38][0] =  data[101]; buffer[38][1] =  data[102]; buffer[38][2] =  data[103];
            buffer[39][0] =  data[102]; buffer[39][1] =  data[103]; buffer[39][2] =  data[104];
            buffer[40][0] =  data[103]; buffer[40][1] =  data[104]; buffer[40][2] =  data[105];
            buffer[41][0] =  data[104]; buffer[41][1] =  data[105]; buffer[41][2] =  data[106];
            buffer[42][0] =  data[105]; buffer[42][1] =  data[106]; buffer[42][2] =  data[107];
            buffer[43][0] =  data[106]; buffer[43][1] =  data[107]; buffer[43][2] =  data[108];
            buffer[44][0] =  data[107]; buffer[44][1] =  data[108]; buffer[44][2] =  data[109];
            buffer[45][0] =  data[108]; buffer[45][1] =  data[109]; buffer[45][2] =  data[110];
            buffer[46][0] =  data[109]; buffer[46][1] =  data[110]; buffer[46][2] =  data[111];
            buffer[47][0] =  data[110]; buffer[47][1] =  data[111]; buffer[47][2] =  data[112];
            buffer[48][0] =  data[111]; buffer[48][1] =  data[112]; buffer[48][2] =  data[113];
            buffer[49][0] =  data[112]; buffer[49][1] =  data[113]; buffer[49][2] =  data[114];
            buffer[50][0] =  data[113]; buffer[50][1] =  data[114]; buffer[50][2] =  data[115];
            buffer[51][0] =  data[114]; buffer[51][1] =  data[115]; buffer[51][2] =  data[116];
            buffer[52][0] =  data[115]; buffer[52][1] =  data[116]; buffer[52][2] =  data[117];
            buffer[53][0] =  data[116]; buffer[53][1] =  data[117]; buffer[53][2] =  data[118];
            buffer[54][0] =  data[117]; buffer[54][1] =  data[118]; buffer[54][2] =  data[119];
            buffer[55][0] =  data[118]; buffer[55][1] =  data[119]; buffer[55][2] =  data[120];
            buffer[56][0] =  data[119]; buffer[56][1] =  data[120]; buffer[56][2] =  data[121];
            buffer[57][0] =  data[120]; buffer[57][1] =  data[121]; buffer[57][2] =  data[122];
            buffer[58][0] =  data[121]; buffer[58][1] =  data[122]; buffer[58][2] =  data[123];
            buffer[59][0] =  data[122]; buffer[59][1] =  data[123]; buffer[59][2] =  data[124];
            buffer[60][0] =  data[123]; buffer[60][1] =  data[124]; buffer[60][2] =  data[125];
            buffer[61][0] =  data[124]; buffer[61][1] =  data[125]; buffer[61][2] =  data[126];
            buffer[62][0] =  data[125]; buffer[62][1] =  data[126]; buffer[62][2] =  data[127];
            buffer[63][0] =  data[126]; buffer[63][1] =  data[127]; buffer[63][2] =  data[128];

        }
        if (partition ==   2) {
            buffer[0][0] =  data[127]; buffer[0][1] =  data[128]; buffer[0][2] =  data[129];
            buffer[1][0] =  data[128]; buffer[1][1] =  data[129]; buffer[1][2] =  data[130];
            buffer[2][0] =  data[129]; buffer[2][1] =  data[130]; buffer[2][2] =  data[131];
            buffer[3][0] =  data[130]; buffer[3][1] =  data[131]; buffer[3][2] =  data[132];
            buffer[4][0] =  data[131]; buffer[4][1] =  data[132]; buffer[4][2] =  data[133];
            buffer[5][0] =  data[132]; buffer[5][1] =  data[133]; buffer[5][2] =  data[134];
            buffer[6][0] =  data[133]; buffer[6][1] =  data[134]; buffer[6][2] =  data[135];
            buffer[7][0] =  data[134]; buffer[7][1] =  data[135]; buffer[7][2] =  data[136];
            buffer[8][0] =  data[135]; buffer[8][1] =  data[136]; buffer[8][2] =  data[137];
            buffer[9][0] =  data[136]; buffer[9][1] =  data[137]; buffer[9][2] =  data[138];
            buffer[10][0] =  data[137]; buffer[10][1] =  data[138]; buffer[10][2] =  data[139];
            buffer[11][0] =  data[138]; buffer[11][1] =  data[139]; buffer[11][2] =  data[140];
            buffer[12][0] =  data[139]; buffer[12][1] =  data[140]; buffer[12][2] =  data[141];
            buffer[13][0] =  data[140]; buffer[13][1] =  data[141]; buffer[13][2] =  data[142];
            buffer[14][0] =  data[141]; buffer[14][1] =  data[142]; buffer[14][2] =  data[143];
            buffer[15][0] =  data[142]; buffer[15][1] =  data[143]; buffer[15][2] =  data[144];
            buffer[16][0] =  data[143]; buffer[16][1] =  data[144]; buffer[16][2] =  data[145];
            buffer[17][0] =  data[144]; buffer[17][1] =  data[145]; buffer[17][2] =  data[146];
            buffer[18][0] =  data[145]; buffer[18][1] =  data[146]; buffer[18][2] =  data[147];
            buffer[19][0] =  data[146]; buffer[19][1] =  data[147]; buffer[19][2] =  data[148];
            buffer[20][0] =  data[147]; buffer[20][1] =  data[148]; buffer[20][2] =  data[149];
            buffer[21][0] =  data[148]; buffer[21][1] =  data[149]; buffer[21][2] =  data[150];
            buffer[22][0] =  data[149]; buffer[22][1] =  data[150]; buffer[22][2] =  data[151];
            buffer[23][0] =  data[150]; buffer[23][1] =  data[151]; buffer[23][2] =  data[152];
            buffer[24][0] =  data[151]; buffer[24][1] =  data[152]; buffer[24][2] =  data[153];
            buffer[25][0] =  data[152]; buffer[25][1] =  data[153]; buffer[25][2] =  data[154];
            buffer[26][0] =  data[153]; buffer[26][1] =  data[154]; buffer[26][2] =  data[155];
            buffer[27][0] =  data[154]; buffer[27][1] =  data[155]; buffer[27][2] =  data[156];
            buffer[28][0] =  data[155]; buffer[28][1] =  data[156]; buffer[28][2] =  data[157];
            buffer[29][0] =  data[156]; buffer[29][1] =  data[157]; buffer[29][2] =  data[158];
            buffer[30][0] =  data[157]; buffer[30][1] =  data[158]; buffer[30][2] =  data[159];
            buffer[31][0] =  data[158]; buffer[31][1] =  data[159]; buffer[31][2] =  data[160];
            buffer[32][0] =  data[159]; buffer[32][1] =  data[160]; buffer[32][2] =  data[161];
            buffer[33][0] =  data[160]; buffer[33][1] =  data[161]; buffer[33][2] =  data[162];
            buffer[34][0] =  data[161]; buffer[34][1] =  data[162]; buffer[34][2] =  data[163];
            buffer[35][0] =  data[162]; buffer[35][1] =  data[163]; buffer[35][2] =  data[164];
            buffer[36][0] =  data[163]; buffer[36][1] =  data[164]; buffer[36][2] =  data[165];
            buffer[37][0] =  data[164]; buffer[37][1] =  data[165]; buffer[37][2] =  data[166];
            buffer[38][0] =  data[165]; buffer[38][1] =  data[166]; buffer[38][2] =  data[167];
            buffer[39][0] =  data[166]; buffer[39][1] =  data[167]; buffer[39][2] =  data[168];
            buffer[40][0] =  data[167]; buffer[40][1] =  data[168]; buffer[40][2] =  data[169];
            buffer[41][0] =  data[168]; buffer[41][1] =  data[169]; buffer[41][2] =  data[170];
            buffer[42][0] =  data[169]; buffer[42][1] =  data[170]; buffer[42][2] =  data[171];
            buffer[43][0] =  data[170]; buffer[43][1] =  data[171]; buffer[43][2] =  data[172];
            buffer[44][0] =  data[171]; buffer[44][1] =  data[172]; buffer[44][2] =  data[173];
            buffer[45][0] =  data[172]; buffer[45][1] =  data[173]; buffer[45][2] =  data[174];
            buffer[46][0] =  data[173]; buffer[46][1] =  data[174]; buffer[46][2] =  data[175];
            buffer[47][0] =  data[174]; buffer[47][1] =  data[175]; buffer[47][2] =  data[176];
            buffer[48][0] =  data[175]; buffer[48][1] =  data[176]; buffer[48][2] =  data[177];
            buffer[49][0] =  data[176]; buffer[49][1] =  data[177]; buffer[49][2] =  data[178];
            buffer[50][0] =  data[177]; buffer[50][1] =  data[178]; buffer[50][2] =  data[179];
            buffer[51][0] =  data[178]; buffer[51][1] =  data[179]; buffer[51][2] =  data[180];
            buffer[52][0] =  data[179]; buffer[52][1] =  data[180]; buffer[52][2] =  data[181];
            buffer[53][0] =  data[180]; buffer[53][1] =  data[181]; buffer[53][2] =  data[182];
            buffer[54][0] =  data[181]; buffer[54][1] =  data[182]; buffer[54][2] =  data[183];
            buffer[55][0] =  data[182]; buffer[55][1] =  data[183]; buffer[55][2] =  data[184];
            buffer[56][0] =  data[183]; buffer[56][1] =  data[184]; buffer[56][2] =  data[185];
            buffer[57][0] =  data[184]; buffer[57][1] =  data[185]; buffer[57][2] =  data[186];
            buffer[58][0] =  data[185]; buffer[58][1] =  data[186]; buffer[58][2] =  data[187];
            buffer[59][0] =  data[186]; buffer[59][1] =  data[187]; buffer[59][2] =  data[188];
            buffer[60][0] =  data[187]; buffer[60][1] =  data[188]; buffer[60][2] =  data[189];
            buffer[61][0] =  data[188]; buffer[61][1] =  data[189]; buffer[61][2] =  data[190];
            buffer[62][0] =  data[189]; buffer[62][1] =  data[190]; buffer[62][2] =  data[191];
            buffer[63][0] =  data[190]; buffer[63][1] =  data[191]; buffer[63][2] =  data[192];

        }
        if (partition ==   3) {
            buffer[0][0] =  data[191]; buffer[0][1] =  data[192]; buffer[0][2] =  data[193];
            buffer[1][0] =  data[192]; buffer[1][1] =  data[193]; buffer[1][2] =  data[194];
            buffer[2][0] =  data[193]; buffer[2][1] =  data[194]; buffer[2][2] =  data[195];
            buffer[3][0] =  data[194]; buffer[3][1] =  data[195]; buffer[3][2] =  data[196];
            buffer[4][0] =  data[195]; buffer[4][1] =  data[196]; buffer[4][2] =  data[197];
            buffer[5][0] =  data[196]; buffer[5][1] =  data[197]; buffer[5][2] =  data[198];
            buffer[6][0] =  data[197]; buffer[6][1] =  data[198]; buffer[6][2] =  data[199];
            buffer[7][0] =  data[198]; buffer[7][1] =  data[199]; buffer[7][2] =  data[200];
            buffer[8][0] =  data[199]; buffer[8][1] =  data[200]; buffer[8][2] =  data[201];
            buffer[9][0] =  data[200]; buffer[9][1] =  data[201]; buffer[9][2] =  data[202];
            buffer[10][0] =  data[201]; buffer[10][1] =  data[202]; buffer[10][2] =  data[203];
            buffer[11][0] =  data[202]; buffer[11][1] =  data[203]; buffer[11][2] =  data[204];
            buffer[12][0] =  data[203]; buffer[12][1] =  data[204]; buffer[12][2] =  data[205];
            buffer[13][0] =  data[204]; buffer[13][1] =  data[205]; buffer[13][2] =  data[206];
            buffer[14][0] =  data[205]; buffer[14][1] =  data[206]; buffer[14][2] =  data[207];
            buffer[15][0] =  data[206]; buffer[15][1] =  data[207]; buffer[15][2] =  data[208];
            buffer[16][0] =  data[207]; buffer[16][1] =  data[208]; buffer[16][2] =  data[209];
            buffer[17][0] =  data[208]; buffer[17][1] =  data[209]; buffer[17][2] =  data[210];
            buffer[18][0] =  data[209]; buffer[18][1] =  data[210]; buffer[18][2] =  data[211];
            buffer[19][0] =  data[210]; buffer[19][1] =  data[211]; buffer[19][2] =  data[212];
            buffer[20][0] =  data[211]; buffer[20][1] =  data[212]; buffer[20][2] =  data[213];
            buffer[21][0] =  data[212]; buffer[21][1] =  data[213]; buffer[21][2] =  data[214];
            buffer[22][0] =  data[213]; buffer[22][1] =  data[214]; buffer[22][2] =  data[215];
            buffer[23][0] =  data[214]; buffer[23][1] =  data[215]; buffer[23][2] =  data[216];
            buffer[24][0] =  data[215]; buffer[24][1] =  data[216]; buffer[24][2] =  data[217];
            buffer[25][0] =  data[216]; buffer[25][1] =  data[217]; buffer[25][2] =  data[218];
            buffer[26][0] =  data[217]; buffer[26][1] =  data[218]; buffer[26][2] =  data[219];
            buffer[27][0] =  data[218]; buffer[27][1] =  data[219]; buffer[27][2] =  data[220];
            buffer[28][0] =  data[219]; buffer[28][1] =  data[220]; buffer[28][2] =  data[221];
            buffer[29][0] =  data[220]; buffer[29][1] =  data[221]; buffer[29][2] =  data[222];
            buffer[30][0] =  data[221]; buffer[30][1] =  data[222]; buffer[30][2] =  data[223];
            buffer[31][0] =  data[222]; buffer[31][1] =  data[223]; buffer[31][2] =  data[224];
            buffer[32][0] =  data[223]; buffer[32][1] =  data[224]; buffer[32][2] =  data[225];
            buffer[33][0] =  data[224]; buffer[33][1] =  data[225]; buffer[33][2] =  data[226];
            buffer[34][0] =  data[225]; buffer[34][1] =  data[226]; buffer[34][2] =  data[227];
            buffer[35][0] =  data[226]; buffer[35][1] =  data[227]; buffer[35][2] =  data[228];
            buffer[36][0] =  data[227]; buffer[36][1] =  data[228]; buffer[36][2] =  data[229];
            buffer[37][0] =  data[228]; buffer[37][1] =  data[229]; buffer[37][2] =  data[230];
            buffer[38][0] =  data[229]; buffer[38][1] =  data[230]; buffer[38][2] =  data[231];
            buffer[39][0] =  data[230]; buffer[39][1] =  data[231]; buffer[39][2] =  data[232];
            buffer[40][0] =  data[231]; buffer[40][1] =  data[232]; buffer[40][2] =  data[233];
            buffer[41][0] =  data[232]; buffer[41][1] =  data[233]; buffer[41][2] =  data[234];
            buffer[42][0] =  data[233]; buffer[42][1] =  data[234]; buffer[42][2] =  data[235];
            buffer[43][0] =  data[234]; buffer[43][1] =  data[235]; buffer[43][2] =  data[236];
            buffer[44][0] =  data[235]; buffer[44][1] =  data[236]; buffer[44][2] =  data[237];
            buffer[45][0] =  data[236]; buffer[45][1] =  data[237]; buffer[45][2] =  data[238];
            buffer[46][0] =  data[237]; buffer[46][1] =  data[238]; buffer[46][2] =  data[239];
            buffer[47][0] =  data[238]; buffer[47][1] =  data[239]; buffer[47][2] =  data[240];
            buffer[48][0] =  data[239]; buffer[48][1] =  data[240]; buffer[48][2] =  data[241];
            buffer[49][0] =  data[240]; buffer[49][1] =  data[241]; buffer[49][2] =  data[242];
            buffer[50][0] =  data[241]; buffer[50][1] =  data[242]; buffer[50][2] =  data[243];
            buffer[51][0] =  data[242]; buffer[51][1] =  data[243]; buffer[51][2] =  data[244];
            buffer[52][0] =  data[243]; buffer[52][1] =  data[244]; buffer[52][2] =  data[245];
            buffer[53][0] =  data[244]; buffer[53][1] =  data[245]; buffer[53][2] =  data[246];
            buffer[54][0] =  data[245]; buffer[54][1] =  data[246]; buffer[54][2] =  data[247];
            buffer[55][0] =  data[246]; buffer[55][1] =  data[247]; buffer[55][2] =  data[248];
            buffer[56][0] =  data[247]; buffer[56][1] =  data[248]; buffer[56][2] =  data[249];
            buffer[57][0] =  data[248]; buffer[57][1] =  data[249]; buffer[57][2] =  data[250];
            buffer[58][0] =  data[249]; buffer[58][1] =  data[250]; buffer[58][2] =  data[251];
            buffer[59][0] =  data[250]; buffer[59][1] =  data[251]; buffer[59][2] =  data[252];
            buffer[60][0] =  data[251]; buffer[60][1] =  data[252]; buffer[60][2] =  data[253];
            buffer[61][0] =  data[252]; buffer[61][1] =  data[253]; buffer[61][2] =  data[254];
            buffer[62][0] =  data[253]; buffer[62][1] =  data[254]; buffer[62][2] =  data[255];
            buffer[63][0] =  data[254]; buffer[63][1] =  data[255]; buffer[63][2] =          0;

        }
    }
};

}

#endif