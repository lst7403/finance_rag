window.TOOLTIP_DEFS = {
    ticker: {
        title: 'Ticker',
        definition: 'Unique stock symbol used to identify a publicly traded company.',
        evaluation: '↑ Quick lookup and comparison',
        filterGuide: 'Search by symbol such as AAPL, MSFT, or NVDA.',
        note: 'Use this to pinpoint a company before comparing fundamentals.',
        tone: 'positive'
    },
    company_name: {
        title: 'Company Name',
        definition: 'The full registered legal or trade name of the corporation.',
        evaluation: '— Identification',
        filterGuide: 'Search by company name or keyword.',
        note: 'Helps identify subsidiaries or conglomerates.',
        tone: 'neutral'
    },
    sector: {
        title: 'Sector',
        definition: 'Broad economic segment in which the company operates (e.g., Technology, Healthcare).',
        evaluation: '— Macro categorization',
        filterGuide: 'Filter by sector name to compare peer groups.',
        note: 'Valuation and margin expectations vary significantly across sectors.',
        tone: 'neutral'
    },
    industry: {
        title: 'Industry',
        definition: 'Specific business category within a sector (e.g., Semiconductors, Medical Devices).',
        evaluation: '— Peer grouping',
        filterGuide: 'Filter by specific industry for direct competitor comparisons.',
        note: 'Companies in the same industry share similar market dynamics and cost structures.',
        tone: 'neutral'
    },
    revenue_growth_1y: {
        title: 'Revenue Growth 1Y',
        definition: 'Year-over-year percentage change in total revenue.',
        evaluation: '↑ Higher is generally better for top-line expansion',
        filterGuide: '≥10% Strong<br>0–10% Moderate<br>&lt;0% Declining',
        note: 'Compare with industry peers and historical revenue trends.',
        tone: 'positive'
    },
    revenue_cagr_3y: {
        title: 'Revenue CAGR 3Y',
        definition: 'Compound annual growth rate of revenue over the past 3 years.',
        evaluation: '↑ Higher indicates sustained long-term growth trajectory',
        filterGuide: '≥15% Strong<br>7–15% Good<br>0–7% Moderate<br>&lt;0% Declining',
        note: 'More reliable than 1Y growth for understanding persistent business growth.',
        tone: 'positive'
    },
    eps_growth_1y: {
        title: 'EPS Growth 1Y',
        definition: 'Year-over-year change in diluted earnings per share.',
        evaluation: '↑ Higher indicates stronger short-term earnings momentum',
        filterGuide: '≥20% Strong<br>0–20% Moderate<br>&lt;0% Declining',
        note: 'EPS growth can be influenced by share buybacks, dilution, and one-off items.',
        tone: 'positive'
    },
    eps_cagr_3y: {
        title: 'EPS CAGR 3Y',
        definition: 'Compound annual growth rate of earnings per share over 3 years.',
        evaluation: '↑ Higher shows sustainable bottom-line compounding power',
        filterGuide: '≥20% Strong<br>10–20% Good<br>0–10% Moderate<br>&lt;0% Declining',
        note: 'Filters out single-year anomalies to reflect underlying profit compounding.',
        tone: 'positive'
    },
    gross_margin: {
        title: 'Gross Margin',
        definition: 'Gross profit divided by revenue. Reflects pricing power and direct production cost efficiency.',
        evaluation: '↑ Higher implies stronger pricing power and moat',
        filterGuide: '≥40% Strong<br>20–40% Moderate<br>&lt;20% Low',
        note: 'Software and luxury goods typically boast much higher gross margins than retail or manufacturing.',
        tone: 'positive'
    },
    operating_margin: {
        title: 'Operating Margin (Op Margin)',
        definition: 'Operating income (EBIT) divided by revenue. Measures core business profitability.',
        evaluation: '↑ Higher shows superior operating leverage and overhead control',
        filterGuide: '≥30% Excellent<br>20–30% Strong<br>10–20% Moderate<br>&lt;10% Weak',
        note: 'High, stable operating margins indicate enduring competitive advantages.',
        tone: 'positive'
    },
    net_margin: {
        title: 'Net Margin',
        definition: 'Net income divided by revenue. Represents bottom-line profit after all expenses, taxes, and interest.',
        evaluation: '↑ Higher indicates superior ultimate profitability',
        filterGuide: '≥15% Strong<br>5–15% Moderate<br>0–5% Weak<br>&lt;0% Negative',
        note: 'Reflects total profitability including capital structure costs.',
        tone: 'positive'
    },
    operating_margin_trend: {
        title: 'Operating Margin Trend',
        definition: 'Change in operating margin over the recent multi-year period.',
        evaluation: '↑ Positive trend indicates margin expansion and operational leverage',
        filterGuide: '&gt;0% Expanding<br>=0% Stable<br>&lt;0% Contracting',
        note: 'Margin expansion combined with revenue growth drives explosive profit acceleration.',
        tone: 'positive'
    },
    roe: {
        title: 'ROE (Return on Equity)',
        definition: 'Net income divided by shareholders equity. Measures return on capital provided by shareholders.',
        evaluation: '↑ Higher indicates more efficient wealth generation for equity owners',
        filterGuide: '≥20% Excellent<br>15–20% Strong<br>8–15% Moderate<br>&lt;8% Weak',
        note: 'Verify with debt levels, as high leverage can artificially inflate ROE.',
        tone: 'positive'
    },
    roic: {
        title: 'ROIC (Return on Invested Capital)',
        definition: 'NOPAT divided by total invested capital (debt + equity). Measures true economic return.',
        evaluation: '↑ Higher indicates superior capital allocation quality above cost of capital',
        filterGuide: '≥15% Excellent<br>12–15% Strong<br>8–12% Moderate<br>&lt;8% Weak',
        note: 'Often considered the single best metric of a durable competitive moat.',
        tone: 'positive'
    },
    fcf_to_net_income: {
        title: 'FCF to Net Income Ratio',
        definition: 'Free cash flow divided by net income. Measures earnings quality and cash conversion.',
        evaluation: '↑ Higher (≥0.8) means accounting earnings convert directly into real cash',
        filterGuide: '≥1.0 Exceptional<br>0.8–1.0 Strong<br>0.5–0.8 Moderate<br>&lt;0.5 Weak/Aggressive',
        note: 'Ratios persistently below 0.5 can signal aggressive revenue recognition or heavy capital intensity.',
        tone: 'positive'
    },
    fcf_margin: {
        title: 'FCF Margin',
        definition: 'Free cash flow divided by revenue. Measures cash generated per dollar of sales.',
        evaluation: '↑ Higher means greater liquidity generation from revenue',
        filterGuide: '≥20% Excellent<br>10–20% Strong<br>5–10% Moderate<br>&lt;5% Weak',
        note: 'High FCF margins allow funding growth, dividends, buybacks, and debt repayment without external capital.',
        tone: 'positive'
    },
    fcf_growth_1y: {
        title: 'FCF Growth 1Y',
        definition: 'Year-over-year percentage change in free cash flow.',
        evaluation: '↑ Positive growth reflects expanding real cash generation',
        filterGuide: '≥20% Strong<br>0–20% Moderate<br>&lt;0% Contracting',
        note: 'Cash flow can fluctuate year to year due to working capital timing and capex cycles.',
        tone: 'positive'
    },
    net_debt_to_ebitda: {
        title: 'Net Debt / EBITDA',
        definition: 'Net debt (Total Debt minus Cash) divided by EBITDA. Standard financial leverage metric.',
        evaluation: '↓ Lower is safer; negative value means net cash position',
        filterGuide: '&lt;0 Net Cash (Very Safe)<br>0–2.0 Safe/Healthy<br>2.0–3.0 Moderate<br>&gt;3.0 High Leverage',
        note: 'Negative numbers indicate that total cash exceeds total debt (net cash balance).',
        tone: 'negative'
    },
    pe_ratio: {
        title: 'P/E Ratio',
        definition: 'Current share price divided by earnings per share (P/E multiple).',
        evaluation: '↓ Lower generally indicates lower valuation multiple',
        filterGuide: '&lt;15 Undervalued/Value<br>15–25 Fair/Moderate<br>25–40 Growth multiple<br>&gt;40 Premium',
        note: 'Always evaluate alongside earnings growth (PEG concept) and margin durability.',
        tone: 'negative'
    }
};
