import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

        # 'Payload Data' 열 제외
        if 'Payload Data' in df.columns:
            df = df.drop(columns=['Payload Data'])

        if 'Timestamp' in df.columns:
            df = df.dropna(subset=['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # 수치형 데이터 결측치를 평균으로 대체
        numerical_cols = ['Packet Length', 'Anomaly Scores']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    col_mean = df[col].mean()
                    df[col] = df[col].fillna(col_mean)

        # 범주형 결측치 처리
        for col in ['Attack Type', 'Severity Level', 'Network Segment', 'Action Taken', 'Protocol',
                    'Source IP Address', 'Destination IP Address']:
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
    """
    if df.empty or 'Action Taken' not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    grouped_counts = df.groupby(group_col)['Action Taken'].agg(
        total_attempts='count',
        blocked_count=lambda x: (x == 'Blocked').sum()
    ).reset_index()

    grouped_counts['Detection Rate'] = grouped_counts['blocked_count'] / grouped_counts['total_attempts']
    grouped_counts['Detection Rate (%)'] = (grouped_counts['Detection Rate'] * 100).round(2)

    return grouped_counts


def safe_get_extremes(df, value_col, label_col):
    """빈 데이터 체크 후 (max_row, min_row) 반환. 없으면 (None, None)."""
    if df.empty or value_col not in df.columns:
        return None, None
    try:
        max_row = df.loc[df[value_col].idxmax()]
        min_row = df.loc[df[value_col].idxmin()]
        return max_row, min_row
    except Exception:
        return None, None


# --- 2. Streamlit 앱 실행 함수 ---
def run_app():
    FILE_PATH = "cybersecurity_attacks.csv"  # 필요시 변경

    data = load_and_preprocess_data(FILE_PATH)

    st.title("🛡️ 사이버시큐리티 공격 요소별 탐지 상관관계 분석")
    st.markdown("##### 💡 목표: 특정 요소 보완을 통한 탐지율 개선 핵심 요소를 식별")
    st.markdown("---")

    if data.empty:
        st.warning("데이터가 비어있거나 로드되지 않았습니다. CSV 경로와 형식을 확인하세요.")
        return

    # --- 사이드바 필터 ---
    st.sidebar.header("분석 데이터 필터")
    filtered_data = data.copy()

    # 공격 유형 필터
    st.sidebar.subheader("1. 공격 유형 선택")
    attack_types = sorted(filtered_data['Attack Type'].unique().tolist()) if 'Attack Type' in filtered_data.columns else []
    selected_attack_type = st.sidebar.multiselect("분석할 공격 유형을 선택하세요:", attack_types, default=attack_types)
    if selected_attack_type:
        filtered_data = filtered_data[filtered_data['Attack Type'].isin(selected_attack_type)]

    # 심각도 필터
    st.sidebar.subheader("2. 심각도 레벨 선택")
    severity_levels = [s for s in ['Low', 'Medium', 'High', 'Unknown'] if s in filtered_data['Severity Level'].unique().tolist()] if 'Severity Level' in filtered_data.columns else []
    selected_severity = st.sidebar.multiselect("분석할 심각도 레벨을 선택하세요:", severity_levels, default=severity_levels)
    if selected_severity:
        filtered_data = filtered_data[filtered_data['Severity Level'].isin(selected_severity)]

    # 네트워크 세그먼트 필터
    st.sidebar.subheader("3. 네트워크 세그먼트 선택")
    network_segments = sorted(filtered_data['Network Segment'].unique().tolist()) if 'Network Segment' in filtered_data.columns else []
    selected_segment = st.sidebar.multiselect("분석할 네트워크 세그먼트를 선택하세요:", network_segments, default=network_segments)
    if selected_segment:
        filtered_data = filtered_data[filtered_data['Network Segment'].isin(selected_segment)]

    # 프로토콜 필터
    st.sidebar.subheader("4. 프로토콜 선택")
    protocols = sorted(filtered_data['Protocol'].unique().tolist()) if 'Protocol' in filtered_data.columns else []
    selected_protocol = st.sidebar.multiselect("분석할 프로토콜을 선택하세요:", protocols, default=protocols)
    if selected_protocol:
        filtered_data = filtered_data[filtered_data['Protocol'].isin(selected_protocol)]

    st.sidebar.markdown("---")

    # 패킷 길이 범위 필터
    st.sidebar.subheader("5. 패킷 길이 범위 필터")
    if 'Packet Length' in filtered_data.columns and not data.empty:
        min_length_data = data['Packet Length'].min()
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

    # --- 핵심 지표 ---
    st.header("1. 핵심 분석 지표 (요약 통계)")
    total_blocked = (filtered_data['Action Taken'] == 'Blocked').sum() if 'Action Taken' in filtered_data.columns else 0
    total_attempts = len(filtered_data)
    overall_detection_rate = (total_blocked / total_attempts) * 100 if total_attempts > 0 else 0

    st.markdown(f"■ 필터링된 데이터 (총 **{total_attempts:,}**개 행)")
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    col_metric_1.metric("● 총 공격 시도 건수", f"{total_attempts:,} 건")
    col_metric_2.metric("● 차단 성공 건수 (탐지 건수)", f"{total_blocked:,} 건")
    col_metric_3.metric("● 평균 탐지율 ('Blocked' / Total)", f"{overall_detection_rate:.2f} %")
    st.markdown("---")

    # --- 분석 선택 ---
    st.header("2. 탐색적 데이터 분석 (상관관계 분석)")
    st.markdown("##### 📌 분석 관점: 네트워크 요소와 탐지율의 상관관계를 중심으로 주요 영향 요인을 분석합니다.")
    analysis_options = [
        '요소별 탐지율 비교 분석 (Protocol, Segment)',
        '시간 흐름 및 탐지 트렌드 분석',
        '심각도 및 조치 결과 교차 분석',
        '수치형 요소 영향 분석 (Packet Length, Anomaly Scores)',
        '공격 주체 및 대상 IP 분석 (Top Talkers)'
    ]
    selected_analysis = st.selectbox("분석 항목을 선택하세요:", analysis_options, key='main_analysis_select')
    st.markdown("---")

    # --- 2-1. 요소별 탐지율 비교 분석 ---
    if selected_analysis == '요소별 탐지율 비교 분석 (Protocol, Segment)':
        st.subheader("🎯 요소별 탐지율 분석: 핵심 보완 요소 식별 (Protocol & Network Segment)")

        col_a, col_b = st.columns(2)

        # 프로토콜별
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

                # 자동 분석 설명 (안전 처리)
                max_row, min_row = safe_get_extremes(protocol_rate_df, 'Detection Rate (%)', 'Protocol')
                if max_row is not None and min_row is not None:
                    diff = (max_row['Detection Rate (%)'] - min_row['Detection Rate (%)'])
                    st.markdown(f"""
                    **📌 분석 해석 (프로토콜)**  
                    - 탐지율이 가장 높은 프로토콜: **{max_row['Protocol']} ({max_row['Detection Rate (%)']}%)**  
                    - 탐지율이 가장 낮은 프로토콜: **{min_row['Protocol']} ({min_row['Detection Rate (%)']}%)**  
                    - 프로토콜별 탐지율 편차: **{diff:.2f}%**
                    """)
                else:
                    st.markdown("프로토콜별 탐지율 정보를 생성할 수 없습니다 (데이터 부족).")

        # 세그먼트별
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

                max_row, min_row = safe_get_extremes(segment_rate_df, 'Detection Rate (%)', 'Network Segment')
                if max_row is not None and min_row is not None:
                    diff = (max_row['Detection Rate (%)'] - min_row['Detection Rate (%)'])
                    st.markdown(f"""
                    **📌 분석 해석 (네트워크 세그먼트)**  
                    - 탐지율이 가장 높은 세그먼트: **{max_row['Network Segment']} ({max_row['Detection Rate (%)']}%)**  
                    - 탐지율이 가장 낮은 세그먼트: **{min_row['Network Segment']} ({min_row['Detection Rate (%)']}%)**   
                    - 세그먼트 간 탐지율 차이: **{diff:.2f}%**
                    """)
                else:
                    st.markdown("네트워크 세그먼트별 탐지율 정보를 생성할 수 없습니다 (데이터 부족).")

    # --- 2-2. 시간 흐름 및 탐지 트렌드 분석 ---
    elif selected_analysis == '시간 흐름 및 탐지 트렌드 분석':
        st.subheader("⏱️ 시간 흐름 및 탐지 트렌드 분석")

        col_time_unit, col_spacer = st.columns([1, 3])
        with col_time_unit:
            time_unit = st.radio("시간 분석 단위:", ('일별', '주별', '월별'), key='time_unit_radio')

        if time_unit == '일별':
            freq = 'D'; title_suffix = "일별 공격 트렌드"
        elif time_unit == '주별':
            freq = 'W'; title_suffix = "주별 공격 트렌드"
        else:
            freq = 'M'; title_suffix = "월별 공격 트렌드"

        # Timestamp가 없는 경우 대비
        if 'Timestamp' not in filtered_data.columns:
            st.info("데이터에 'Timestamp' 컬럼이 없어 시간 흐름 분석이 불가능합니다.")
        else:
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
                yaxis='y1',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>총 공격 건수: %{y:,} 건<extra></extra>'
            ))
            fig_line_combined.add_trace(go.Scatter(
                x=trend_data['Timestamp'],
                y=trend_data['Detection Rate (%)'],
                name='탐지율 (%)',
                mode='lines+markers',
                yaxis='y2',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>탐지율: %{y:.2f}%<extra></extra>'
            ))
            fig_line_combined.update_layout(
                title=f'총 공격 건수 및 탐지율 변화 ({title_suffix})',
                xaxis_title="시간",
                yaxis=dict(title='총 공격 건수 (좌측)', side='left', showgrid=False, rangemode='nonnegative'),
                yaxis2=dict(title='탐지율 (%) (우측)', overlaying='y', side='right', range=[0, 100], showgrid=True, dtick=10, ticksuffix='%'),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_line_combined, use_container_width=True)

            # 자동 분석 설명 (안전 처리)
            if not trend_data.empty:
                try:
                    max_rate = trend_data.loc[trend_data['Detection Rate (%)'].idxmax()]
                    min_rate = trend_data.loc[trend_data['Detection Rate (%)'].idxmin()]
                    peak_attacks = trend_data.loc[trend_data['Total Attacks'].idxmax()]
                    st.markdown(f"""
                    **📌 분석 해석 (시간 흐름)**  
                    - 가장 탐지율이 높았던 시점: **{pd.to_datetime(max_rate['Timestamp']).date()} ({max_rate['Detection Rate (%)']}%)**  
                    - 탐지율이 가장 낮았던 시점: **{pd.to_datetime(min_rate['Timestamp']).date()} ({min_rate['Detection Rate (%)']}%)**  
                    - 공격이 가장 많이 발생한 시점: **{pd.to_datetime(peak_attacks['Timestamp']).date()} ({int(peak_attacks['Total Attacks']):,}건)**  
                    - 다양한 요소에 따라 지속적으로 바뀌므로, 예측이 불가능 합니다
                    """)
                except Exception:
                    st.markdown("시간 흐름 분석을 위한 요약 정보를 생성할 수 없습니다.")

    # --- 2-3. 심각도 및 조치 결과 교차 분석 ---
    elif selected_analysis == '심각도 및 조치 결과 교차 분석':
        st.subheader("🔥 심각도 및 조치 결과 교차 상관 분석")

        if 'Severity Level' not in filtered_data.columns or 'Action Taken' not in filtered_data.columns:
            st.info("심각도 또는 조치 결과 컬럼이 없어 교차 분석을 실행할 수 없습니다.")
        else:
            cross_tab = pd.crosstab(filtered_data['Severity Level'], filtered_data['Action Taken'], normalize=False)
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

                # 자동 분석 설명
                try:
                    most_common = cross_tab.sum(axis=1).idxmax()
                    least_common = cross_tab.sum(axis=1).idxmin()
                    st.markdown(f"""
                    **📌 분석 해석 (심각도 × 조치)**  
                    - 가장 많이 발생한 심각도: **{most_common}**  
                    - 가장 적게 발생한 심각도: **{least_common}**  
                    - 아쉽게도 데이터셋 자체가 심각도를 비슷하도록 만들어졌기 때문에, 의도와는 다르게 어느 요소를 보강해야 할지 알 수 없었습니다
                    """)
                except Exception:
                    st.markdown("교차 분석 요약 정보를 생성할 수 없습니다.")

    # --- 2-4. 수치형 요소 영향 분석 ---
    elif selected_analysis == '수치형 요소 영향 분석 (Packet Length, Anomaly Scores)':
        st.subheader("📏 수치형 요소 영향 분석")
        col_length, col_anomaly = st.columns(2)
        action_data = filtered_data[filtered_data['Action Taken'].isin(['Blocked', 'Ignored', 'Logged'])].copy() if 'Action Taken' in filtered_data.columns else filtered_data.copy()

        with col_length:
            st.markdown("##### 패킷 길이(`Packet Length`) 분포와 조치 결과 비교")
            if not action_data.empty and 'Packet Length' in action_data.columns:
                fig_length_dist = px.box(
                    action_data,
                    x='Action Taken' if 'Action Taken' in action_data.columns else None,
                    y='Packet Length',
                    color='Action Taken' if 'Action Taken' in action_data.columns else None,
                    title='패킷 길이와 조치 결과의 분포 비교 (Log Scale)',
                    labels={'Packet Length': '패킷 길이 (Log Scale)'},
                    log_y=True,
                    category_orders={"Action Taken": ['Blocked', 'Logged', 'Ignored']}
                )
                st.plotly_chart(fig_length_dist, use_container_width=True)

                # 간단 요약
                try:
                    median_by_action = action_data.groupby('Action Taken')['Packet Length'].median().sort_values(ascending=False) if 'Action Taken' in action_data.columns else pd.Series()
                    if not median_by_action.empty:
                        top_action = median_by_action.index[0]
                        st.markdown(f"**📌 분석 해석 (패킷 길이)**  - 중앙값이 가장 큰 조치: **{top_action}** (중앙값: {median_by_action.iloc[0]:.2f})")
                except Exception:
                    st.markdown("패킷 길이 분포의 요약을 생성할 수 없습니다.")
            else:
                st.info("표시할 패킷 길이 데이터가 없습니다.")

        with col_anomaly:
            st.markdown("##### 비정상 점수(`Anomaly Scores`) 분포와 조치 결과 비교")
            if not action_data.empty and 'Anomaly Scores' in action_data.columns:
                fig_anomaly_dist = px.violin(
                    action_data,
                    x='Action Taken' if 'Action Taken' in action_data.columns else None,
                    y='Anomaly Scores',
                    color='Action Taken' if 'Action Taken' in action_data.columns else None,
                    box=True,
                    title='비정상 점수와 조치 결과의 분포 비교',
                    labels={'Anomaly Scores': '비정상 점수'},
                    category_orders={"Action Taken": ['Blocked', 'Logged', 'Ignored']}
                )
                st.plotly_chart(fig_anomaly_dist, use_container_width=True)

                # 간단 요약
                try:
                    mean_by_action = action_data.groupby('Action Taken')['Anomaly Scores'].mean().sort_values(ascending=False) if 'Action Taken' in action_data.columns else pd.Series()
                    if not mean_by_action.empty:
                        top_action = mean_by_action.index[0]
                        st.markdown(f"**📌 분석 해석 (비정상 점수)**  - 평균 비정상 점수가 가장 높은 조치: **{top_action}** (평균: {mean_by_action.iloc[0]:.2f})")
                except Exception:
                    st.markdown("비정상 점수 분포의 요약을 생성할 수 없습니다.")
            else:
                st.info("표시할 비정상 점수 데이터가 없습니다.")

    # --- 2-5. 공격 주체 및 대상 IP 분석 ---
    elif selected_analysis == '공격 주체 및 대상 IP 분석 (Top Talkers)':
        st.subheader("👤 공격 주체 및 대상 IP 분석 (Top Talkers)")

        top_n = st.slider("표시할 상위 IP 개수 (N):", 5, 20, 10)
        col_source, col_dest = st.columns(2)

        with col_source:
            st.markdown(f"##### 상위 {top_n}개 공격 시도 IP (`Source IP Address`)")
            if 'Source IP Address' in filtered_data.columns:
                top_source_ips = filtered_data['Source IP Address'].value_counts().nlargest(top_n).reset_index()
                top_source_ips.columns = ['Source IP Address', 'Count']

                fig_source = px.bar(
                    top_source_ips,
                    x='Count',
                    y='Source IP Address',
                    orientation='h',
                    title=f'Top {top_n} 공격 시도 IP',
                    color='Count',
                    text_auto=True
                )
                fig_source.update_yaxes(categoryorder='total ascending')
                st.plotly_chart(fig_source, use_container_width=True)

                if not top_source_ips.empty:
                    top_ip = top_source_ips.iloc[0]
                    st.markdown(f"""
                    **📌 분석 해석 (공격 주체)**  
                    공격자들이 IP를 통한 추적을 막기 위해서 같은 IP를 되도록 사용하지 않기 때문에,
                    전부 모든 IP가 1번식 사용된 것을 알 수 있다.
                    """)
                else:
                    st.markdown("상위 공격 주체 정보를 생성할 수 없습니다.")
            else:
                st.info("'Source IP Address' 컬럼이 없어 Top Talkers(발신자) 분석을 수행할 수 없습니다.")

    st.markdown("---")
    # --- 3. 원본 데이터 테이블 ---
    st.header("📄 3. 분석에 사용된 데이터 미리보기 (전처리 완료)")
    st.info(f"현재 총 {len(filtered_data):,} 건의 데이터가 필터링되었습니다.")
    st.dataframe(filtered_data.head(500), use_container_width=True)


if __name__ == "__main__":
    run_app()
