import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter

# --- 0. Streamlit 애플리케이션 설정 ---
st.set_page_config(
    page_title="사이버시큐리티 공격 요소별 탐지 상관관계 분석",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    지정된 경로에서 CSV 파일을 로드하고 프로젝트 기획에 따라 전처리합니다.
    """
    try:
        df = pd.read_csv(file_path)

        # 기획안 반영: 'Payload Data'열 제외
        if 'Payload Data' in df.columns:
            df = df.drop(columns=['Payload Data'])

        if 'Timestamp' in df.columns:
            df = df.dropna(subset=['Timestamp']) # 먼저 NaN을 제거해야 to_datetime이 에러 없이 작동
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # 기획안 반영: 수치형 데이터 결측치를 해당 요소의 평균값으로 계산해서 처리
        numerical_cols = ['Packet Length', 'Anomaly Scores']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    col_mean = df[col].mean()
                    df[col] = df[col].fillna(col_mean)
        
        # 범주형 데이터 결측치 처리 (Unknown으로 대체)
        for col in ['Attack Type', 'Severity Level', 'Network Segment', 'Action Taken', 'Protocol', 'Source IP Address', 'Destination IP Address']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        required_cols = ['Timestamp', 'Source IP Address', 'Destination IP Address', 'Protocol', 'Packet Length', 
                         'Attack Type', 'Severity Level', 'Action Taken', 'Anomaly Scores', 'Network Segment']
        
        df = df[[col for col in required_cols if col in df.columns]].copy()
        
        return df

    except FileNotFoundError:
        st.error(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해 주세요: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로딩 및 전처리 중 오류 발생: {e}")
        return pd.DataFrame()


def calculate_detection_rate_by_group(df, group_col):
    """
    주어진 컬럼을 기준으로 그룹별 탐지율 ('Blocked' / Total)을 계산합니다.
    (기획안 반영: 탐지 = 'Blocked' 액션으로 정의)
    """
    if df.empty or 'Action Taken' not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    grouped_counts = df.groupby(group_col)['Action Taken'].agg(
        total_attempts='count',
        blocked_count=lambda x: (x == 'Blocked').sum()
    ).reset_index()

    # 탐지율 (Blocked / Total) 계산
    grouped_counts['Detection Rate'] = grouped_counts['blocked_count'] / grouped_counts['total_attempts']
    grouped_counts['Detection Rate (%)'] = (grouped_counts['Detection Rate'] * 100).round(2)
    
    return grouped_counts

# --- 2. Streamlit 앱 실행 함수 ---
def run_app():
    """메인 Streamlit 애플리케이션 로직"""

    FILE_PATH = "cybersecurity_attacks.csv" 

    # 데이터 로드 (원본 데이터)
    data = load_and_preprocess_data(FILE_PATH)

    # --- 대시보드 제목 섹션 (프로젝트 기획안 반영) ---
    st.title("🛡️ 사이버시큐리티 공격 요소별 탐지 상관관계 분석")
    st.markdown("##### 💡 목표: 특정 요소 보완을 통한 탐지율 개선 핵심 요소를 식별")
    
    st.markdown("---")


    if data.empty:
        return

    # --- 사이드바 (Filters) ---
    st.sidebar.header("분석 데이터 필터")
    
    filtered_data = data.copy()

    # 1. 공격 유형 필터
    st.sidebar.subheader("1. 공격 유형 선택")
    attack_types = sorted(filtered_data['Attack Type'].unique().tolist())
    selected_attack_type = st.sidebar.multiselect("분석할 공격 유형을 선택하세요:", attack_types, default=attack_types)
    if selected_attack_type:
        filtered_data = filtered_data[filtered_data['Attack Type'].isin(selected_attack_type)]

    # 2. 심각도 레벨 필터
    st.sidebar.subheader("2. 심각도 레벨 선택")
    severity_levels = [s for s in ['Low', 'Medium', 'High', 'Unknown'] if s in filtered_data['Severity Level'].unique().tolist()]
    selected_severity = st.sidebar.multiselect("분석할 심각도 레벨을 선택하세요:", severity_levels, default=severity_levels)
    if selected_severity:
        filtered_data = filtered_data[filtered_data['Severity Level'].isin(selected_severity)]

    # 3. 네트워크 세그먼트 필터
    st.sidebar.subheader("3. 네트워크 세그먼트 선택")
    network_segments = sorted(filtered_data['Network Segment'].unique().tolist())
    selected_segment = st.sidebar.multiselect("분석할 네트워크 세그먼트를 선택하세요:", network_segments, default=network_segments)
    if selected_segment:
        filtered_data = filtered_data[filtered_data['Network Segment'].isin(selected_segment)]

    # 4. 프로토콜 필터
    st.sidebar.subheader("4. 프로토콜 선택")
    protocols = sorted(filtered_data['Protocol'].unique().tolist())
    selected_protocol = st.sidebar.multiselect("분석할 프로토콜을 선택하세요:", protocols, default=protocols)
    if selected_protocol:
        filtered_data = filtered_data[filtered_data['Protocol'].isin(selected_protocol)]
    
    st.sidebar.markdown("---")

    # 5. 패킷 길이 범위 필터 (수치형)
    st.sidebar.subheader("5. 패킷 길이 범위 필터")
    if 'Packet Length' in filtered_data.columns and not data.empty:
        min_length_data = data['Packet Length'].min() # 전체 데이터의 min/max 기준
        max_length_data = data['Packet Length'].max()
        length_range = st.sidebar.slider(
            "패킷 길이 범위:",
            float(min_length_data), 
            float(max_length_data), 
            (float(min_length_data), float(max_length_data))
        )
        filtered_data = filtered_data[
            (filtered_data['Packet Length'] >= length_range[0]) & 
            (filtered_data['Packet Length'] <= length_range[1])
        ]
    else:
        st.sidebar.info("필터링할 'Packet Length' 데이터가 없습니다.")
        
    st.sidebar.markdown("---")
    st.sidebar.caption(f"필터링된 데이터 (총 **{len(filtered_data):,}**개 행)")

    if filtered_data.empty:
        st.warning("선택된 필터 조건에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
        return

    # --- 메인 영역 (핵심 지표 및 요약) ---
    st.header("1. 핵심 분석 지표 (요약 통계)")
    
    # 지표 계산
    total_blocked = (filtered_data['Action Taken'] == 'Blocked').sum()
    total_attempts = len(filtered_data)
    # 기획안 반영: 탐지율 = Blocked / Total
    overall_detection_rate = (total_blocked / total_attempts) * 100 if total_attempts > 0 else 0
    
    st.markdown(f"■ 필터링된 데이터 (총 **{total_attempts:,}**개 행)")
    
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    
    col_metric_1.metric("● 총 공격 시도 건수", f"{total_attempts:,} 건")
    col_metric_2.metric("● 차단 성공 건수 (탐지 건수)", f"{total_blocked:,} 건")
    col_metric_3.metric("● 평균 탐지율 ('Blocked' / Total)", f"{overall_detection_rate:.2f} %")
    
    st.markdown("---")

    # --- 메인 영역 (시각화 분석 항목 선택) ---
    st.header("2. 탐색적 데이터 분석 (상관관계 분석)")
    st.markdown("##### 📌 분석 관점: 네트워크 요소와 탐지율의 상관관계를 중심으로 주요 영향 요인을 분석합니다.")

    analysis_options = [
        '요소별 탐지율 비교 분석 (Protocol, Segment)',
        '시간 흐름 및 탐지 트렌드 분석',
        '심각도 및 조치 결과 교차 분석',
        '수치형 요소 영향 분석 (Packet Length, Anomaly Scores)',
        '공격 주체 및 대상 IP 분석 (Top Talkers)'
    ]

    selected_analysis = st.selectbox(
        "분석 항목을 선택하세요:",
        analysis_options,
        key='main_analysis_select'
    )
    
    st.markdown("---")
    
    # --- 2-1. 요소별 탐지율 비교 분석 (기획안의 핵심 관점) ---
    if selected_analysis == '요소별 탐지율 비교 분석 (Protocol, Segment)':
        st.subheader("🎯 요소별 탐지율 분석: 핵심 보완 요소 식별 (Protocol & Network Segment)")

        col_a, col_b = st.columns(2)

        with col_a:
            protocol_rate_df = calculate_detection_rate_by_group(filtered_data, 'Protocol')
            st.markdown("##### 프로토콜(`Protocol`)별 탐지율 (낮을수록 보완 필요)")
            if not protocol_rate_df.empty:
                fig_proto_corr = px.bar(
                    protocol_rate_df.sort_values(by='Detection Rate (%)', ascending=False),
                    x='Protocol',
                    y='Detection Rate (%)',
                    color='Protocol',
                    title='프로토콜별 탐지율 비교',
                    height=400,
                    hover_data=['total_attempts', 'blocked_count']
                )
                fig_proto_corr.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_proto_corr, use_container_width=True)

        with col_b:
            segment_rate_df = calculate_detection_rate_by_group(filtered_data, 'Network Segment')
            st.markdown("##### 네트워크 세그먼트(`Network Segment`)별 탐지율 (낮을수록 보완 필요)")
            if not segment_rate_df.empty:
                fig_segment_corr = px.bar(
                    segment_rate_df.sort_values(by='Detection Rate (%)', ascending=False),
                    x='Network Segment',
                    y='Detection Rate (%)',
                    color='Network Segment',
                    title='네트워크 세그먼트별 탐지율 비교',
                    height=400,
                    hover_data=['total_attempts', 'blocked_count']
                )
                st.plotly_chart(fig_segment_corr, use_container_width=True)
                
    # --- 2-2. 시간 흐름 및 탐지 트렌드 분석 ---
    elif selected_analysis == '시간 흐름 및 탐지 트렌드 분석':
        st.subheader("⏱️ 시간 흐름 및 탐지 트렌드 분석")

        col_time_unit, col_spacer = st.columns([1, 3])

        with col_time_unit:
            time_unit = st.radio(
                "시간 분석 단위:",
                ('일별', '주별', '월별'),
                key='time_unit_radio' 
            )

        if time_unit == '일별':
            freq = 'D'
            title_suffix = "일별 공격 트렌드"
        elif time_unit == '주별':
            freq = 'W'
            title_suffix = "주별 공격 트렌드"
        else:
            freq = 'M'
            title_suffix = "월별 공격 트렌드"

        trend_data = filtered_data.set_index('Timestamp').resample(freq).size().reset_index(name='Total Attacks')
        blocked_data = filtered_data[filtered_data['Action Taken'] == 'Blocked'].set_index('Timestamp').resample(freq).size().reset_index(name='Blocked Attacks')
        
        trend_data = pd.merge(trend_data, blocked_data, on='Timestamp', how='left').fillna(0)
        
        trend_data['Detection Rate (%)'] = np.where(
            trend_data['Total Attacks'] > 0,
            (trend_data['Blocked Attacks'] / trend_data['Total Attacks'] * 100).round(2),
            0
        )
        
        fig_line_combined = go.Figure()
        
        fig_line_combined.add_trace(go.Bar(
            x=trend_data['Timestamp'],
            y=trend_data['Total Attacks'],
            name='총 공격 건수',
            marker_color='rgba(31, 119, 180, 0.6)',
            yaxis='y1',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + '총 공격 건수: %{y:,} 건<extra></extra>' 
        ))

        fig_line_combined.add_trace(go.Scatter(
            x=trend_data['Timestamp'],
            y=trend_data['Detection Rate (%)'],
            name='탐지율 (%)',
            mode='lines+markers',
            line=dict(color='red', width=2),
            yaxis='y2',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + '탐지율: %{y:.2f}%<extra></extra>' 
        ))

        fig_line_combined.update_layout(
            title=f'총 공격 건수 및 탐지율 변화 ({title_suffix})',
            xaxis_title="시간",
            yaxis=dict(title='총 공격 건수 (좌측)', side='left', showgrid=False, rangemode='nonnegative'),
            yaxis2=dict(title='탐지율 (%) (우측)', overlaying='y', side='right', range=[0, 100], showgrid=True, dtick=10, ticksuffix='%'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_line_combined, use_container_width=True)

    # --- 2-3. 심각도 및 조치 결과 교차 분석 ---
    elif selected_analysis == '심각도 및 조치 결과 교차 분석':
        st.subheader("🔥 심각도 및 조치 결과 교차 상관 분석")

        cross_tab = pd.crosstab(
            filtered_data['Severity Level'], 
            filtered_data['Action Taken'], 
            normalize=False
        )
        
        current_severity_order = [s for s in ['High', 'Medium', 'Low', 'Unknown'] if s in cross_tab.index]
        cross_tab = cross_tab.reindex(current_severity_order, fill_value=0).fillna(0) 

        st.markdown("##### 심각도 레벨과 조치 결과 간의 관계 (Heatmap)")
        if not cross_tab.empty:
            fig_heatmap = px.imshow(
                cross_tab,
                text_auto=True, 
                aspect="auto",
                color_continuous_scale=px.colors.sequential.Teal,
                title='심각도 레벨과 조치 결과 간의 관계 (공격 건수)',
                labels={'x': '조치 결과 (Action Taken)', 'y': '심각도 레벨 (Severity Level)'}
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- 2-4. 수치형 요소 영향 분석 ---
    elif selected_analysis == '수치형 요소 영향 분석 (Packet Length, Anomaly Scores)':
        st.subheader("📏 수치형 요소 영향 분석")
        
        col_length, col_anomaly = st.columns(2)
        # Action Taken이 존재하는 유효한 데이터만 사용 (Ignored/Logged/Blocked)
        action_data = filtered_data[filtered_data['Action Taken'].isin(['Blocked', 'Ignored', 'Logged'])].copy()

        with col_length:
            st.markdown("##### 패킷 길이(`Packet Length`) 분포와 조치 결과 비교")
            if not action_data.empty:
                fig_length_dist = px.box(
                    action_data,
                    x='Action Taken',
                    y='Packet Length',
                    color='Action Taken',
                    title='패킷 길이와 조치 결과의 분포 비교 (Log Scale)',
                    labels={'Packet Length': '패킷 길이 (Log Scale)'},
                    log_y=True,
                    category_orders={"Action Taken": ['Blocked', 'Logged', 'Ignored']}
                )
                st.plotly_chart(fig_length_dist, use_container_width=True)

        with col_anomaly:
            st.markdown("##### 비정상 점수(`Anomaly Scores`) 분포와 조치 결과 비교")
            if not action_data.empty:
                fig_anomaly_dist = px.violin(
                    action_data,
                    x='Action Taken',
                    y='Anomaly Scores',
                    color='Action Taken',
                    box=True,
                    title='비정상 점수와 조치 결과의 분포 비교',
                    labels={'Anomaly Scores': '비정상 점수'},
                    category_orders={"Action Taken": ['Blocked', 'Logged', 'Ignored']}
                )
                st.plotly_chart(fig_anomaly_dist, use_container_width=True)

    # --- 2-5. 공격 주체 및 대상 IP 분석 ---
    elif selected_analysis == '공격 주체 및 대상 IP 분석 (Top Talkers)':
        st.subheader("👤 공격 주체 및 대상 IP 분석 (Top Talkers)")

        top_n = st.slider("표시할 상위 IP 개수 (N):", 5, 20, 10)
        col_source, col_dest = st.columns(2)

        with col_source:
            st.markdown(f"##### 상위 {top_n}개 공격 시도 IP (`Source IP Address`)")
            top_source_ips = filtered_data['Source IP Address'].value_counts().nlargest(top_n).reset_index()
            top_source_ips.columns = ['Source IP Address', 'Count']
            
            fig_source = px.bar(
                top_source_ips,
                x='Count',
                y='Source IP Address',
                orientation='h',
                title=f'Top {top_n} 공격 시도 IP',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues,
                text_auto=True
            )
            fig_source.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_source, use_container_width=True)

        with col_dest:
            st.markdown(f"##### 상위 {top_n}개 공격 대상 IP (`Destination IP Address`)")
            top_dest_ips = filtered_data['Destination IP Address'].value_counts().nlargest(top_n).reset_index()
            top_dest_ips.columns = ['Destination IP Address', 'Count']

            fig_dest = px.bar(
                top_dest_ips,
                x='Count',
                y='Destination IP Address',
                orientation='h',
                title=f'Top {top_n} 공격 대상 IP',
                color='Count',
                color_continuous_scale=px.colors.sequential.Greens,
                text_auto=True
            )
            fig_dest.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_dest, use_container_width=True)
    
    st.markdown("---")
    
    # --- 3. 원본 데이터 테이블 (프로젝트 수행 참고용) ---
    st.header("📄 3. 분석에 사용된 데이터 미리보기 (전처리 완료)")
    st.info(f"현재 총 {len(filtered_data):,} 건의 데이터가 필터링되었습니다.")
    st.dataframe(filtered_data.head(500), use_container_width=True)

if __name__ == "__main__":
    run_app()