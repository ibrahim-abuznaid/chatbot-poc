import React from 'react';
import { Select, MenuItem } from '@mui/material';

interface QueryTranslationSelectorProps {
  qtt: string;
  setQtt: (value: string) => void;
}

const QueryTranslationSelector = ({ qtt, setQtt }: QueryTranslationSelectorProps) => {
  return (
    <Select
      value={qtt}
      onChange={(e) => setQtt(e.target.value)}
      sx={{ minWidth: 120 }}
    >
      <MenuItem value="basic">Basic</MenuItem>
      <MenuItem value="fusion">Fusion</MenuItem>
      <MenuItem value="decomposition">Decomposition</MenuItem>
      <MenuItem value="hyde">HyDE</MenuItem>
    </Select>
  );
};

export default QueryTranslationSelector; 